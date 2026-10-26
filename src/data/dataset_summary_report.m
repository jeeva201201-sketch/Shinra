function summary_text = dataset_summary_report()
% DATASET_SUMMARY_REPORT Generate a comprehensive audit of all project datasets
% Produces human-readable summary text and writes report to results/reports/.

    cfg = shinra_config();
    inventory = scan_datasets();
    
    [idrid_train, idrid_test] = load_idrid();
    [drive_train, drive_test] = load_drive();
    aptos_table = load_aptos();
    
    lines = {};
    lines{end+1} = '========================================================================';
    lines{end+1} = '                    SHINRA DATASET AUDIT & INVENTORY                    ';
    lines{end+1} = sprintf('  Generated on: %s', char(datetime('now')));
    lines{end+1} = '========================================================================';
    lines{end+1} = '';
    
    % IDRiD Section
    lines{end+1} = '1. IDRiD DATASET (Indian Diabetic Retinopathy Image Dataset)';
    lines{end+1} = sprintf('   • Total Training Images: %d', height(idrid_train));
    lines{end+1} = sprintf('   • Total Testing Images:  %d', height(idrid_test));
    lines{end+1} = sprintf('   • Total Ground Truth DR Labels: %d', height(idrid_train) + height(idrid_test));
    lines{end+1} = '   • DR Severity Class Distribution (Train Split):';
    for g = 0:4
        cnt = sum(idrid_train.DR_Grade == g);
        pct = (cnt / height(idrid_train)) * 100;
        lines{end+1} = sprintf('       - Grade %d (%-16s): %3d (%5.1f%%)', g, cfg.severity.class_names{g+1}, cnt, pct);
    end
    lines{end+1} = '   • DR Severity Class Distribution (Test Split):';
    for g = 0:4
        cnt = sum(idrid_test.DR_Grade == g);
        pct = (cnt / height(idrid_test)) * 100;
        lines{end+1} = sprintf('       - Grade %d (%-16s): %3d (%5.1f%%)', g, cfg.severity.class_names{g+1}, cnt, pct);
    end
    ref_train = sum(idrid_train.Referable_DR == 1);
    ref_test  = sum(idrid_test.Referable_DR == 1);
    lines{end+1} = sprintf('   • Referable DR (Grade >= 2): Train = %d (%.1f%%), Test = %d (%.1f%%)', ...
        ref_train, (ref_train/height(idrid_train))*100, ref_test, (ref_test/height(idrid_test))*100);
    lines{end+1} = sprintf('   • Pixel-Level Lesion Annotations Present: %s', yes_no_str(inventory.idrid.has_lesion_masks));
    lines{end+1} = '';
    
    % DRIVE Section
    lines{end+1} = '2. DRIVE DATASET (Digital Retinal Images for Vessel Extraction)';
    lines{end+1} = sprintf('   • Training Images: %d (.tif)', height(drive_train));
    lines{end+1} = sprintf('   • Training Manual Vessel Masks: %d (.gif)', sum(~cellfun(@isempty, drive_train.ManualMaskPath)));
    lines{end+1} = sprintf('   • Training Retinal FOV Masks:   %d (.gif)', sum(~cellfun(@isempty, drive_train.FOVMaskPath)));
    lines{end+1} = sprintf('   • Testing Images: %d (.tif)', height(drive_test));
    lines{end+1} = sprintf('   • Testing Retinal FOV Masks:    %d (.gif)', sum(~cellfun(@isempty, drive_test.FOVMaskPath)));
    lines{end+1} = '';
    
    % APTOS Section
    lines{end+1} = '3. APTOS 2019 DATASET';
    lines{end+1} = sprintf('   • Total Fundus Images Found: %d (.png)', height(aptos_table));
    lines{end+1} = sprintf('   • Labels File Present: %s', yes_no_str(inventory.aptos.has_labels));
    lines{end+1} = '   • Role in SHINRA: Unlabelled rural throughput, batch screening, and quality-gating evaluation.';
    lines{end+1} = '========================================================================';
    
    summary_text = strjoin(lines, '\n');
    
    % Write to results/reports/dataset_inventory_report.txt
    report_file = fullfile(cfg.paths.reports, 'dataset_inventory_report.txt');
    fid = fopen(report_file, 'w');
    if fid ~= -1
        fprintf(fid, '%s\n', summary_text);
        fclose(fid);
    end
end

function s = yes_no_str(b)
    if b
        s = 'YES';
    else
        s = 'NO';
    end
end
