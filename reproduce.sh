#!/usr/bin/env bash
# reproduce.sh - runs the AttentionRSA scripts for one group of networks, or for all three
#
# Groups (results in results/<folder>/)
#   radius             UNet-Radius-mini, 10 copies                                          rsa_radius
#   radius-attention   Attention UNet-Radius-mini, 10 copies                                rsa_radius_attention
#   starlight          UNet-Starlight and Attention UNet-Starlight with a lens output,      rsa_starlight
#                      5 copies each
#
# Steps
#   1. simulate.py           the simulated set (once, for all groups)
#   2. train_copies.py       the copies, trained on the simulated set (--init-seed 1 ... N)
#   3. finetune_radius.py    radius groups: every copy applied to the real lenses and fine-tuned on their measured
#                            Einstein radii; the same network trained on them from random weights (seeds 1 ... N)
#      finetune_lens.py      starlight group: every copy fine-tuned on EUCLID-SAGE-1k-D (copy 1 searches the learning
#                            rate, copies 2 ... N use its choices); the same networks trained on 1k-D from random
#                            weights (seeds 1 ... N)
#   4. compute_rdms.py       the RDMs of every layer of every network
#   5. compare_rdms.py       the comparisons of the RDMs (radius and radius-attention together, if both are done)
#
# Folders: EUCLID_SAGE_DATA_DIR (working folder, default: current folder), EUCLID_SAGE_BENCHMARK_DIR (benchmarks/ of
# the EUCLID-SAGE repository), MEASURED_RADII (default: modeling_lens_mass.csv in the working folder; radius groups).
#
# Usage
#   bash reproduce.sh all                  all groups
#   bash reproduce.sh radius               one group
#   bash reproduce.sh all --quick          short test run (2 copies, few epochs; results in folders ending in _quick)
set -euo pipefail

GROUPS_TO_RUN="${1:-all}"
QUICK=""
if [ "${2:-}" = "--quick" ]; then QUICK="--quick"; fi
SUFFIX="${QUICK:+_quick}"
DATA_FOLDER="${EUCLID_SAGE_DATA_DIR:-.}"
RESULTS="$DATA_FOLDER/results"
RDMS="$DATA_FOLDER/rdms"
MEASURED_RADII="${MEASURED_RADII:-$DATA_FOLDER/modeling_lens_mass.csv}"

BENCHMARK_FOLDER="${EUCLID_SAGE_BENCHMARK_DIR:-$DATA_FOLDER/benchmarks}"
if [ ! -f "$BENCHMARK_FOLDER/EUCLID-SAGE-1k-D_VIS.csv" ]; then
    echo "EUCLID-SAGE benchmarks not found in $BENCHMARK_FOLDER; set EUCLID_SAGE_BENCHMARK_DIR (see README.md)." >&2
    exit 1
fi
export EUCLID_SAGE_BENCHMARK_DIR="$BENCHMARK_FOLDER"

case "$GROUPS_TO_RUN" in
    all) GROUP_LIST="radius radius-attention starlight" ;;
    radius|radius-attention|starlight) GROUP_LIST="$GROUPS_TO_RUN" ;;
    *) echo "Unknown group: $GROUPS_TO_RUN (radius, radius-attention, starlight or all)" >&2; exit 1 ;;
esac

# 1. Simulated set
if [ -n "$QUICK" ]; then
    SIMULATED_SET="$DATA_FOLDER/radius_training_200.npz"; SIMULATION_SIZE="--n-lens 100 --n-nonlens 100"
else
    SIMULATED_SET="$DATA_FOLDER/radius_training_10000.npz"; SIMULATION_SIZE=""
fi
if [ ! -f "$SIMULATED_SET" ]; then
    python simulate.py --seed 42 --noise-seed 1 $SIMULATION_SIZE --out "$SIMULATED_SET"
fi

for GROUP in $GROUP_LIST; do
    case "$GROUP" in
        radius)           FOLDER=rsa_radius;           COPIES=10; NETWORKS="radius-mini" ;;
        radius-attention) FOLDER=rsa_radius_attention; COPIES=10; NETWORKS="attention-radius-mini" ;;
        starlight)        FOLDER=rsa_starlight;        COPIES=5;  NETWORKS="starlight attention-starlight" ;;
    esac
    if [ -n "$QUICK" ]; then COPIES=2; fi
    if [ "$GROUP" != "starlight" ] && [ ! -f "$MEASURED_RADII" ]; then
        echo "Group $GROUP skipped: $MEASURED_RADII not found (modeling_lens_mass.csv, Zenodo record 15025832)." >&2
        continue
    fi
    echo "=== Group $GROUP: $NETWORKS, $COPIES copies each"

    # 2. Copies trained on the simulated set
    for K in $(seq 1 $COPIES); do
        for NETWORK in $NETWORKS; do
            COPY_FOLDER="$RESULTS/$FOLDER/simulations/$NETWORK-init$(printf %02d $K)$SUFFIX"
            case "$NETWORK" in
                radius-mini)           OPTIONS="--model radius-mini" ;;
                attention-radius-mini) OPTIONS="--model radius-mini --attention" ;;
                starlight)             OPTIONS="--model starlight --lens-output" ;;
                attention-starlight)   OPTIONS="--model starlight --lens-output --attention" ;;
            esac
            if [ ! -f "$COPY_FOLDER/summary.json" ]; then
                python train_copies.py --data "$SIMULATED_SET" $OPTIONS --init-seed $K --out "$COPY_FOLDER" $QUICK
            fi
        done
    done

    # 3. Fine-tuning on real stamps, and training on them from random weights
    SEEDS=$(seq -s " " 1 $COPIES)
    if [ "$GROUP" != "starlight" ]; then
        COPY_NETWORKS=""
        for K in $(seq 1 $COPIES); do
            COPY="$NETWORKS-init$(printf %02d $K)"
            COPY_NETWORKS="$COPY_NETWORKS $COPY=$RESULTS/$FOLDER/simulations/$COPY$SUFFIX/best_unet.pt"
        done
        python finetune_radius.py --measured "$MEASURED_RADII" --networks $COPY_NETWORKS --seeds 1 --no-random \
            --out-name "$FOLDER/einstein_radius_real" $QUICK
        FIRST="$NETWORKS-init01=$RESULTS/$FOLDER/simulations/$NETWORKS-init01$SUFFIX/best_unet.pt"
        python finetune_radius.py --measured "$MEASURED_RADII" --networks "$FIRST" --seeds $SEEDS --no-pretrained \
            --out-name "$FOLDER/einstein_radius_real_random_weights" $QUICK
    else
        for NETWORK in $NETWORKS; do
            for K in $(seq 1 $COPIES); do
                COPY="$NETWORK-init$(printf %02d $K)"
                REUSE=""
                if [ "$K" -gt 1 ]; then
                    REUSE="--learning-rates-from $RESULTS/$FOLDER/fine_tuning_1k_D/$NETWORK-init01$SUFFIX"
                fi
                python finetune_lens.py --checkpoint "$RESULTS/$FOLDER/simulations/$COPY$SUFFIX/best_unet.pt" \
                    --init pretrained --seeds 1 --out-name "$FOLDER/fine_tuning_1k_D/$COPY" $REUSE $QUICK
            done
            python finetune_lens.py --checkpoint "$RESULTS/$FOLDER/simulations/$NETWORK-init01$SUFFIX/best_unet.pt" \
                --init random --seeds $SEEDS --out-name "$FOLDER/random_weights_1k_D/$NETWORK" $QUICK
        done
    fi

    # 4. RDMs of every layer of every network
    python compute_rdms.py --results "$RESULTS/$FOLDER" --simulated "$SIMULATED_SET" --out "$RDMS/$FOLDER$SUFFIX"
done

# 5. Comparisons
if [ -d "$RDMS/rsa_radius$SUFFIX" ] && [ -d "$RDMS/rsa_radius_attention$SUFFIX" ]; then
    python compare_rdms.py --rdms "$RDMS/rsa_radius$SUFFIX" "$RDMS/rsa_radius_attention$SUFFIX" \
        --simulated "$SIMULATED_SET" --out "$DATA_FOLDER/comparisons/radius$SUFFIX"
else
    for FOLDER in rsa_radius rsa_radius_attention; do
        if [ -d "$RDMS/$FOLDER$SUFFIX" ]; then
            python compare_rdms.py --rdms "$RDMS/$FOLDER$SUFFIX" --simulated "$SIMULATED_SET" \
                --out "$DATA_FOLDER/comparisons/$FOLDER$SUFFIX"
        fi
    done
fi
if [ -d "$RDMS/rsa_starlight$SUFFIX" ]; then
    python compare_rdms.py --rdms "$RDMS/rsa_starlight$SUFFIX" --simulated "$SIMULATED_SET" \
        --out "$DATA_FOLDER/comparisons/starlight$SUFFIX"
fi
