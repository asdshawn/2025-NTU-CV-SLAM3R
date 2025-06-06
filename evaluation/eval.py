import os
from argparse import ArgumentParser
from utils import load_points_from_ply, accuracy, completion

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-rec", "--reconstruct", dest="rec_path",
                        help="Path to the reconstruction directory containing .ply files",
                        default=f"{os.curdir}/../results/recon_points/"
                        required=True)
    parser.add_argument("-gt", "--ground_truth", dest="gt_path",
                        help="Path to the ground truth .ply file",
                        default=f"{os.curdir}/../data/gt_points/",
                        required=True)
    parser.add_argument("-b", "--bonus", dest="bonus",
                        help="Bonus evaluation mode",
                        action='store_true', default=False)
    
    rec_path = parser.parse_args().sequence_path
    gt_path = parser.parse_args().ply_path
    bonus = parser.parse_args().bonus
    
    if bonus:
        seq = ['chess-seq-03', 'chess-sparse-seq-05', 'fire-seq-03', 'fire-sparse-seq-04',
               'heads-seq-01', 'office-seq-02', 'office-seq-06', 'office-seq-07', 'office-seq-09',
               'pumpkin-seq-01', 'pumpkin-sparse-seq-07', 'redkitchen-seq-03', 'redkitchen-seq-04',
               'redkitchen-seq-06', 'redkitchen-seq-12', 'redkitchen-seq-14', 'stairs-seq-01',
               'stairs-sparse-seq-04']
    else:
        seq = ['chess-seq-03', 'fire-seq-03', 'heads-seq-01', 'office-seq-02', 'office-seq-06',
               'office-seq-07', 'office-seq-09', 'pumpkin-seq-01', 'redkitchen-seq-03',
               'redkitchen-seq-04', 'redkitchen-seq-06', 'redkitchen-seq-12', 'redkitchen-seq-14',
               'stairs-seq-01']
    
    acc_list = []
    acc_median_list = []
    comp_list = []
    comp_median_list = []
    
    for s in seq:
        rec_points = load_points_from_ply(f'{rec_path}/{s}.ply')
        gt_points = load_points_from_ply(f'{gt_path}/{s}.ply')
        acc, acc_median = accuracy(gt_points, rec_points)
        comp, comp_median = completion(gt_points, rec_points)
        print(f'[{s}] acc: {acc:.4f}, acc_median: {acc_median:.4f}, comp: {comp:.4f}, comp_median: {comp_median:.4f}')
        
        acc_list.append(acc)
        acc_median_list.append(acc_median)
        comp_list.append(comp)
        comp_median_list.append(comp_median)
    
    # 計算平均值
    avg_acc = sum(acc_list) / len(acc_list)
    avg_acc_median = sum(acc_median_list) / len(acc_median_list)
    avg_comp = sum(comp_list) / len(comp_list)
    avg_comp_median = sum(comp_median_list) / len(comp_median_list)
    
    # 輸出平均值
    print(f'\nAverage acc: {avg_acc:.4f}')
    print(f'Average acc_median: {avg_acc_median:.4f}')
    print(f'Average comp: {avg_comp:.4f}')
    print(f'Average comp_median: {avg_comp_median:.4f}')