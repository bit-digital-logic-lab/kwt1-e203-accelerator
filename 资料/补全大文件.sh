#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kwt_dir="$root_dir/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1"
paper_dir="$root_dir/论文"
mkdir -p "$kwt_dir" "$paper_dir"

fetch() {
  local url="$1" dest="$2" tmp
  tmp="${dest}.part"
  curl --noproxy '*' --http1.1 -fL --retry 5 --retry-all-errors --connect-timeout 20 -C - -o "$tmp" "$url"
  mv "$tmp" "$dest"
}

check_git_blob() {
  local dest="$1" expected="$2" actual
  actual="$(git hash-object "$dest")"
  if [[ "$actual" != "$expected" ]]; then
    printf '校验失败: %s\n' "$dest" >&2
    exit 1
  fi
}

weight_file="$kwt_dir/best_weights.data-00000-of-00001"
if [[ ! -f "$weight_file" ]]; then
  python3 "$root_dir/下载KWT1权重.py"
fi
check_git_blob "$weight_file" 9c38f9653119383f7a303a94689c85a2a1c79b6b

paper_file="$paper_dir/Keyword_Transformer_2104.00769.pdf"
if [[ ! -f "$paper_file" ]]; then
  fetch 'https://arxiv.org/pdf/2104.00769' "$paper_file"
fi
if [[ "$(head -c 4 "$paper_file")" != '%PDF' ]]; then
  printf 'PDF 文件头校验失败: %s\n' "$paper_file" >&2
  exit 1
fi

if [[ "${1:-}" == '--dataset' ]]; then
  dataset_dir="$root_dir/数据集"
  mkdir -p "$dataset_dir"
  dataset_file="$dataset_dir/speech_commands_v0.02.tar.gz"
  if [[ ! -f "$dataset_file" ]]; then
    fetch 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz' "$dataset_file"
  fi
  if [[ "$(md5sum "$dataset_file" | cut -d ' ' -f 1)" != '6b74f3901214cb2c2934e98196829835' ]]; then
    printf '数据集 MD5 校验失败: %s\n' "$dataset_file" >&2
    exit 1
  fi
  gzip -t "$dataset_file"
fi

printf '已下载并校验指定文件。\n'
