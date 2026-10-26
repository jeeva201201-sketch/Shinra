function test_fusion_reliability()
% TEST_FUSION_RELIABILITY Unit tests for multi-modal evidence fusion and reliability triage
    setup_SHINRA();
    sev_res = struct('probabilities', [0.05, 0.10, 0.55, 0.20, 0.10], 'p_referable', 0.85, 'confidence', 0.55);
    lesion_ev = struct('ma_count', 12, 'ma_area', 180, 'he_count', 6, 'he_area_fraction', 0.004, ...
        'ex_count', 15, 'ex_area_fraction', 0.008, 'dme_risk', 2.0, 'nv_score', 0.15);
    vess_res = struct('morphometry', struct('vessel_density', 0.095, 'mean_tortuosity', 1.20));
    q_rep = struct('score', 78.0, 'quality_class', 'GOOD');

    ev_vec = extractEvidenceVector(sev_res, lesion_ev, vess_res, q_rep);
    fus_res = fuseEvidence(ev_vec, []);
    rel_res = computeReliability(sev_res.confidence, q_rep.score, fus_res.agreement_score);
    dec_res = makeScreeningDecision(q_rep, fus_res, rel_res);

    assert(fus_res.p_referable_fused >= 0.50, 'Severe lesion burden should yield referable fusion.');
    assert(strcmp(dec_res.final_decision, 'REFERABLE DR'), 'Decision should be REFERABLE DR.');
    fprintf('test_fusion_reliability: PASSED\n');
end
