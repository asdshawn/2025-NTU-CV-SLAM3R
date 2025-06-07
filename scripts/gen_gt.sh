#!/bin/bash

inp_dir="../data/7SCENES/"
# 確保輸出目錄存在
if [ ! -d "../results/gt_points/" ]; then
    mkdir -p ../results/gt_points/
fi
# 輸出目錄
outp_dir="../results/gt_points/"

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

echo "Starting ground true PLY Generation for sequences..."

# 遍歷 SEQ 陣列的鍵 (例如 "chess", "fire", ...)
for folder_name in "${!SEQ[@]}"; do
    # 取得與鍵對應的值 (例如 "seq-03" 或 "seq-02 seq-06 ...")
    seq_str="${SEQ[$folder_name]}"

    # 將序列字串分割成個別序列 (如果有多個)
    # IFS (Internal Field Separator) 暫時設定為空格，以便正確分割
    IFS=' ' read -r -a sequences <<< "$seq_str"

    # 遍歷該資料夾名稱下的所有序列
    for seq_val in "${sequences[@]}"; do
        seq_path="${inp_dir}${folder_name}/test/${seq_val}/"\

        echo "Processing outp_directory: ${seq_path}"

        python3 ../evaluation/seq2ply.py -inp ${seq_path} -outp ${outp_dir}/"${folder_name}-${seq_val}.ply"

        echo ""
    done
done

echo "PLY Generation finished."