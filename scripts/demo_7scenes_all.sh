#!/bin/bash

declare -A SEQ

SEQ=(
    ["chess"]="seq-03 sparse-seq-05"
    ["fire"]="seq-03 sparse-seq-04"
    ["heads"]="seq-01"
    ["office"]="seq-02 seq-06 seq-07 seq-09"
    ["pumpkin"]="seq-01 sparse-seq-07"
    ["redkitchen"]="seq-03 seq-04 seq-06 seq-12 seq-14"
    ["stairs"]="seq-01 sparse-seq-04"
)

# 遍歷 SEQ 陣列的鍵 (例如 "chess", "fire", ...)
for folder_name in "${!SEQ[@]}"; do
    # 取得與鍵對應的值 (例如 "seq-03" 或 "seq-02 seq-06 ...")
    seq_str="${SEQ[$folder_name]}"

    # 將序列字串分割成個別序列 (如果有多個)
    # IFS (Internal Field Separator) 暫時設定為空格，以便正確分割
    IFS=' ' read -r -a sequences <<< "$seq_str"

    # 遍歷該資料夾名稱下的所有序列
    for seq_val in "${sequences[@]}"; do
        seq_path="data/7SCENES/${folder_name}/test/${seq_val}/"

        bash scripts/demo_7scenes.sh ${seq_path} &
    done
done

wait

echo "All sequences processed successfully."