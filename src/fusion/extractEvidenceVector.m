function [evidence_vec, feature_names] = extractEvidenceVector(severity_res, lesion_evidence, vessel_res, quality_rep)
% EXTRACTEVIDENCEVECTOR Construct Multi-Modal Retinal Evidence Feature Vector
% Combines deep CNN probabilities, quantified lesion burden (MAs, Hemorrhages, Exudates),
% vascular morphometry, and image quality metrics into a unified 16-dimensional evidence representation.

    feature_names = {
        'P_No_DR', 'P_Mild_NPDR', 'P_Mod_NPDR', 'P_Sev_NPDR', 'P_PDR', ...
        'P_Referable_CNN', ...
        'MA_Count', 'MA_Area_Total', ...
        'HE_Count', 'HE_Area_Fraction', ...
        'EX_Count', 'EX_Area_Fraction', 'DME_Risk_Score', ...
        'Vessel_Density', 'Vessel_Tortuosity', ...
        'Quality_Score'
    };

    evidence_vec = zeros(1, numel(feature_names));

    % 1. CNN Probabilities (Features 1 to 6)
    if isfield(severity_res, 'probabilities') && numel(severity_res.probabilities) == 5
        evidence_vec(1:5) = severity_res.probabilities;
    else
        evidence_vec(1:5) = [0.6, 0.15, 0.12, 0.08, 0.05];
    end
    evidence_vec(6) = severity_res.p_referable;

    % 2. Microaneurysm Burden (Features 7 & 8)
    if isfield(lesion_evidence, 'ma_count')
        evidence_vec(7) = min(50, lesion_evidence.ma_count);
        evidence_vec(8) = min(1000, lesion_evidence.ma_area);
    end

    % 3. Hemorrhage Burden (Features 9 & 10)
    if isfield(lesion_evidence, 'he_count')
        evidence_vec(9)  = min(50, lesion_evidence.he_count);
        evidence_vec(10) = min(0.10, lesion_evidence.he_area_fraction);
    end

    % 4. Exudate Burden & DME Risk (Features 11, 12, 13)
    if isfield(lesion_evidence, 'ex_count')
        evidence_vec(11) = min(50, lesion_evidence.ex_count);
        evidence_vec(12) = min(0.10, lesion_evidence.ex_area_fraction);
        evidence_vec(13) = lesion_evidence.dme_risk;
    end

    % 5. Vascular Morphometry (Features 14 & 15)
    if isfield(vessel_res, 'morphometry')
        evidence_vec(14) = vessel_res.morphometry.vessel_density;
        evidence_vec(15) = vessel_res.morphometry.mean_tortuosity;
    else
        evidence_vec(14) = 0.08;
        evidence_vec(15) = 1.0;
    end

    % 6. Image Quality (Feature 16)
    if isfield(quality_rep, 'score')
        evidence_vec(16) = quality_rep.score;
    else
        evidence_vec(16) = 75.0;
    end
end
