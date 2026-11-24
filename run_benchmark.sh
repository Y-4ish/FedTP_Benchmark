#!/usr/bin/env bash
# The server uses Bash 4.2, where nounset rejects even initialized empty arrays.
set -eo pipefail

# Unified federated benchmark entry point. Run from any working directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zzh/anaconda3/envs/ft/bin/python}"
MODEL_SPEC=""
DATASET_SPEC="PeMS03,PeMS04,PeMSD7,PeMS08"
FEATURE_SPEC="flow,speed,occ"
NUM_CLIENTS=4
DEVICE="cuda:0"
DRY_RUN=0
NODE_CLIENTS=0
EXTRA_ARGS=()

MODELS=(TwoMGTCN ASTGAT STGCN UFCL UFCL_GWN DyHSL FedTSE FedOSTC pFedCTP FedTPS FedGODE FedGTP FedSTN CNFGNN FCGCN FCFedGCN FGNNEH FedMetro STFAM FedSTG FUELS FedGRU Fed4TP SFL_RNN FedmSSA REFOL TDLR SEDLR TDLR_SEDLR ISTGNN)

usage() {
  cat <<'EOF'
Usage: bash run_benchmark.sh --model MODEL|all [options] [-- extra fate_main.py arguments]

Options:
  --model NAME[,NAME...]  One or more models; use all for every listed model.
  --datasets NAME[,NAME...]  Default: PeMS03,PeMS04,PeMSD7,PeMS08
  --features NAME[,NAME...]  Default: flow,speed,occ
  --num-clients N          Default: 4
  --device DEVICE          Default: cuda:0
  --python PATH            Default: /home/zzh/anaconda3/envs/ft/bin/python
  --dry-run                Print commands without launching training.
  --node-clients           One client per sensor, using four FATE workers.
                           Supports CNFGNN, FedOSTC, FedGODE, FedSTG, FedTSE,
                           FUELS, REFOL, SFL and TDLR/SEDLR (plain PeMS).
  --list-models            List model names and aliases.
  --help                   Show this help.

Every run uses at most 1000 federated rounds, 50-round early stopping,
batch_size=64, t_in=12, t_out=3, Adam, and a constant learning rate of 0.01.
Local training follows each method where compatible with these settings.
Extra arguments cannot override these benchmark settings.

Examples:
  bash run_benchmark.sh --model FedMetro --datasets PeMS04 --features flow
  bash run_benchmark.sh --model FedTSE,FedMetro --dry-run
  bash run_benchmark.sh --model all --datasets PeMS04 --features flow
EOF
}

list_models() {
  printf '%s\n' "${MODELS[@]}"
  printf '%s\n' 'Aliases: 2MGTCN=TwoMGTCN FedAGAT=ASTGAT STAGCN_EC=STGCN Fed4TP=FedGRU/fed4tp SFL=SFL_RNN UFCL=STGCN/ufcl FC-FedGCN=FCFedGCN T-ISTGNN=ISTGNN source-only (no target transfer)'
}

while (($#)); do
  case "$1" in
    --model|--datasets|--features|--num-clients|--device|--python)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --model) MODEL_SPEC="$2" ;;
        --datasets) DATASET_SPEC="$2" ;;
        --features) FEATURE_SPEC="$2" ;;
        --num-clients) NUM_CLIENTS="$2" ;;
        --device) DEVICE="$2" ;;
        --python) PYTHON_BIN="$2" ;;
      esac
      shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --node-clients) NODE_CLIENTS=1; shift ;;
    --list-models) list_models; exit 0 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$MODEL_SPEC" ]] || { usage >&2; exit 2; }
[[ "$NUM_CLIENTS" =~ ^[1-9][0-9]*$ ]] || { echo 'num-clients must be a positive integer' >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 2; }

# Reject benchmark overrides even when argparse would otherwise accept them.
for arg in "${EXTRA_ARGS[@]}"; do
  case "$arg" in
    --model|--model=*|--dataset_name|--dataset_name=*|--feature_type|--feature_type=*|--num_clients|--num_clients=*|--batch_size|--batch_size=*|--t_in|--t_in=*|--t_out|--t_out=*|--epochs|--epochs=*|--lr|--lr=*|--stfam_lstm_lr|--stfam_lstm_lr=*|--stfam_cnn_lr|--stfam_cnn_lr=*|--fedmssa_impute_lr|--fedmssa_impute_lr=*|--benchmark_fixed_lr|--benchmark_fixed_lr=*|--patience|--patience=*|--early_stop_patience|--early_stop_patience=*|--fedmssa_patience|--fedmssa_patience=*|--tdlr_patience|--tdlr_patience=*|--tdlr_stream_rounds|--tdlr_stream_rounds=*|--time_window_num|--time_window_num=*|--result_root|--result_root=*)
      echo "Benchmark argument is fixed by this script: $arg" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a REQUESTED_MODELS <<< "$MODEL_SPEC"
IFS=',' read -r -a DATASETS <<< "$DATASET_SPEC"
IFS=',' read -r -a FEATURES <<< "$FEATURE_SPEC"
if [[ "$MODEL_SPEC" == all ]]; then REQUESTED_MODELS=("${MODELS[@]}"); fi
((${#DATASETS[@]} && ${#FEATURES[@]})) || { echo 'Dataset and feature lists cannot be empty' >&2; exit 2; }
for feature in "${FEATURES[@]}"; do
  case "$feature" in flow|speed|occ) ;; *) echo "Unknown feature: $feature" >&2; exit 2 ;; esac
done

canonical_model() {
  case "$1" in
    2MGTCN) echo TwoMGTCN ;;
    FedAGAT) echo ASTGAT ;;
    STAGCN_EC|STAGCN-EC) echo STGCN ;;
    Fed4TP) echo Fed4TP ;;
    SFT|SFL) echo SFL_RNN ;;
    FC-FedGCN) echo FCFedGCN ;;
    T-ISTGNN) echo ISTGNN ;;
    *) echo "$1" ;;
  esac
}

export WANDB_MODE=offline WANDB_DISABLED=true PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
failures=0

for requested in "${REQUESTED_MODELS[@]}"; do
  name="$(canonical_model "$requested")"
  model="$name"
  trainer_mode=""
  hidden_dim=64 lr=0.01 epochs=1000 seed=42 local_epochs=1 loss_func=mae
  model_args=()

  case "$name" in
    TwoMGTCN) ;;
    ASTGAT) hidden_dim=64 ;;
    STGCN) ;;
    UFCL) model=STGCN; trainer_mode=ufcl; model_args+=(--lambda_1 5.0 --lambda_2 0.1 --temperature 0.5) ;;
    UFCL_GWN) trainer_mode=ufcl; model_args+=(--lambda_1 5.0 --lambda_2 0.1 --temperature 0.5) ;;
    DyHSL) model_args+=(--wd 0.0001 --dyhsl_num_backbone_layers 7 --dyhsl_num_head_layers 2 --dyhsl_num_hyper_edge 32 --dyhsl_winsize 2 --dyhsl_dropout 0 --dyhsl_scales 1,2,3,4,6,12) ;;
    FedTSE) hidden_dim=100 ;;
    FedOSTC) hidden_dim=64; local_epochs=5 ;;
    pFedCTP) hidden_dim=16; local_epochs=150; loss_func=mse; model_args+=(--target_epochs 50) ;;
    FedTPS) model_args+=(--fedtps_k 2) ;;
    FedGODE) loss_func=smoothl1 ;;
    FedGTP) local_epochs=2; model_args+=(--poly_k 4 --node_emb_dim 2) ;;
    FedSTN) ;;
    CNFGNN) ;;
    FCGCN) loss_func=mse; model_args+=(--fcgcn_adj_residual_alpha 0) ;;
    FCFedGCN) loss_func=mse ;;
    FGNNEH) hidden_dim=32; model_args+=(--fgnneh_n_components 8 --fgnneh_alpha 0.1 --fgnneh_beta 0.05) ;;
    FedMetro) seed=12
      model_args+=(--node_emb_dim 6 --poly_k 2 --fedmetro_act_k 4 --fedmetro_grad_clip 0 --fedmetro_d_D 32 --fedmetro_mask_beta 0.66 --fedmetro_mask_gamma -0.1 --fedmetro_mask_zeta 1.1 --lambda_reg 0.001 --sparsification_threshold 0.8) ;;
    STFAM) local_epochs=10; loss_func=mse; model_args+=(--stfam_embed_dim 64 --tsvd_dim 10 --num_clusters 3 --stfam_lstm_lr 0.01 --stfam_cnn_lr 0.01) ;;
    FedSTG) hidden_dim=32; model_args+=(--node_emb_dim 32 --fedstg_alpha 0.01 --fedstg_beta 0.5 --fedstg_sigma 0.0 --fedstg_kappa 0.0) ;;
    FUELS) model_args+=(--fuels_dr 128 --fuels_c 6 --fuels_q 6 --fuels_tau 0.02 --fuels_rho 5.0 --fuels_beta_percentile 50.0 --fuels_intra_weight 1.0 --fuels_aug_noise_std 0.01 --fuels_aug_mask_ratio 0.10 --fuels_aug_shift_prob 0.50) ;;
    FedGRU) hidden_dim=50; loss_func=mse ;;
    Fed4TP) model=FedGRU; trainer_mode=fed4tp; hidden_dim=100; epochs=200; local_epochs=10; loss_func=mse; model_args+=(--fed4tp_gd_use --fed4tp_mg_use --time_window_num 5 --rho 0.1) ;;
    SFL_RNN) trainer_mode=sfl; model_args+=(--sfl_lambda 0.1 --sfl_m_steps 1 --sfl_gamma 0.01) ;;
    FedmSSA) local_epochs=10; loss_func=mse; model_args+=(--fedmssa_page_length 50 --fedmssa_rank 0 --fedmssa_sv_scale 2.0 --fedmssa_lstm_layers 2 --fedmssa_dropout 0.0 --fedmssa_missing_ratio 0.0 --fedmssa_impute_rounds 100 --fedmssa_impute_local_steps 50 --fedmssa_impute_lr 0.01 --fedmssa_consensus_weight 1.0 --fedmssa_ortho_weight 1.0 --fedmssa_diag_weight 0.2 --fedmssa_server_momentum 0.0) ;;
    REFOL) hidden_dim=128; local_epochs=5; loss_func=mse; model_args+=(--kl_threshold 0.0003 --refol_aggregation gcn) ;;
    TDLR|SEDLR|TDLR_SEDLR) hidden_dim=128; local_epochs=5; loss_func=mae; model_args+=(--tdlr_stream_rounds 1000 --tdlr_patience 50 --sedlr_occ_threshold 30 --sedlr_warmup_rounds 0 --sedlr_cooldown_rounds 0) ;;
    ISTGNN) loss_func=mse ;;
    *) echo "Unsupported model: $requested (use --list-models)" >&2; exit 2 ;;
  esac

  if ((NODE_CLIENTS)); then
    case "$name" in CNFGNN|FedOSTC|FedGODE|FedSTG|FedTSE|FUELS|REFOL|SFL_RNN|TDLR|SEDLR|TDLR_SEDLR) ;; *)
      echo "Node clients are not implemented for $name" >&2; exit 2 ;;
    esac
    [[ "$NUM_CLIENTS" == 4 ]] || { echo 'Node client mode requires four workers' >&2; exit 2; }
    model_args+=(--node_clients)
  fi

  for dataset in "${DATASETS[@]}"; do
    [[ -n "$dataset" ]] || { echo 'Empty dataset name' >&2; exit 2; }
    for feature in "${FEATURES[@]}"; do
      result_root="$ROOT/logs/benchmark/in12_out3/$name"
      command=("$PYTHON_BIN" -u run/fate_main.py
        --model "$model" --dataset_name "$dataset" --feature_type "$feature"
        --num_clients "$NUM_CLIENTS" --device "$DEVICE"
        --epochs "$epochs" --lr "$lr" --benchmark_fixed_lr --hidden_dim "$hidden_dim"
        --local_epochs "$local_epochs" --seed "$seed"
        --normalizer "$(case "$name" in FCGCN|FCFedGCN) echo max01 ;; *) echo std ;; esac)" --loss_func "$loss_func"
        --train_ratio 0.7 --val_ratio 0.1 --test_ratio 0.2
        "${model_args[@]}" "${EXTRA_ARGS[@]}")
      if [[ -n "$trainer_mode" ]]; then command+=(--trainer_mode "$trainer_mode"); fi
      command+=(--batch_size 64 --t_in 12 --t_out 3 --result_root "$result_root")
      printf '[benchmark] %s %s %s: ' "$name" "$dataset" "$feature"
      printf '%q ' "${command[@]}"
      printf '\n'
      if ((DRY_RUN)); then continue; fi
      mkdir -p "$result_root"
      timestamp="$(date +%Y%m%d_%H%M%S)"
      log_file="$result_root/${dataset}_${feature}_${timestamp}.log"
      echo "[benchmark] log: $log_file"
      if "${command[@]}" > "$log_file" 2>&1; then
        echo "[benchmark] completed: $name $dataset $feature"
      else
        status=$?
        echo "[benchmark] failed ($status): $name $dataset $feature; see $log_file" >&2
        failures=$((failures + 1))
      fi
    done
  done
done

if ((failures)); then
  echo "[benchmark] $failures run(s) failed" >&2
  exit 1
fi
