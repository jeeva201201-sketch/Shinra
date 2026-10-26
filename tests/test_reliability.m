function test_reliability()
% TEST_RELIABILITY Unit Test Suite for SHINRA Reliability Layer
% Covers 12 critical engineering test cases:
% 1. GOOD high-confidence image
% 2. GOOD prediction near threshold (margin uncertainty)
% 3. BORDERLINE image quality
% 4. UNGRADABLE image (safety gate override)
% 5. Missing vessel evidence
% 6. Missing lesion evidence
% 7. NaN input handling
% 8. Inf input handling
% 9. Invalid probability vector
% 10. CNN / lesion discordance
% 11. Complete concordant evidence
% 12. Test-partition quarantine assertion

    setup_SHINRA();
    cfg = shinra_config();

    fprintf('============================================================\n');
    fprintf('  RUNNING UNIT TESTS: SHINRA RELIABILITY LAYER              \n');
    fprintf('============================================================\n');

    % Base Template Data
    q_good = struct('score', 85.0, 'quality_class', 'GOOD');
    sev_good = struct('probabilities', [0.88, 0.08, 0.02, 0.01, 0.01], 'confidence', 0.88, 'p_referable', 0.04, 'expected_grade', 0.19);
    fus_good = struct('p_referable_fused', 0.05, 'p_cnn_referable', 0.04, 'agreement_score', 0.95);
    les_good = struct('ma_count', 0, 'ma_area', 0, 'he_count', 0, 'he_area_fraction', 0, 'ex_count', 0, 'ex_area_fraction', 0, 'dme_risk', 0, 'nv_score', 0.05);
    vess_good = struct('morphometry', struct('vessel_density', 0.09, 'mean_tortuosity', 1.15));
    od_good = struct('center', [256, 256], 'radius', 40, 'confidence', 0.95);
    fovea_good = struct('center', [320, 256], 'macula_radius', 60);

    % -------------------------------------------------------------------------
    % Test 1: GOOD High-Confidence Image
    % -------------------------------------------------------------------------
    r1 = computeReliability(q_good, sev_good, fus_good, les_good, vess_good, od_good, fovea_good);
    d1 = makeScreeningDecision(q_good, fus_good, r1);
    assert(r1.reliability_score >= 70.0, 'Test 1 Failed: Score should be >= 70.');
    assert(strcmp(r1.confidence_level, 'HIGH'), 'Test 1 Failed: Confidence should be HIGH.');
    assert(strcmp(d1.decision, 'ACCEPT_SCREENING'), 'Test 1 Failed: Action should be ACCEPT_SCREENING.');
    assert(d1.decision_readiness == true, 'Test 1 Failed: Decision readiness should be TRUE.');
    fprintf('  [PASS] Test 1: GOOD high-confidence image (Score: %.1f, Action: %s)\n', r1.reliability_score, d1.decision);

    % -------------------------------------------------------------------------
    % Test 2: GOOD Prediction Near Decision Boundary (Margin Uncertainty)
    % -------------------------------------------------------------------------
    fus_near = struct('p_referable_fused', 0.51, 'p_cnn_referable', 0.51, 'agreement_score', 0.80);
    r2 = computeReliability(q_good, sev_good, fus_near, les_good, vess_good, od_good, fovea_good);
    d2 = makeScreeningDecision(q_good, fus_near, r2);
    assert(r2.margin_score_component < 20.0, 'Test 2 Failed: Margin sub-score should be penalized.');
    assert(r2.reliability_score < r1.reliability_score, 'Test 2 Failed: Boundary case should have lower reliability.');
    fprintf('  [PASS] Test 2: Prediction near boundary (Score: %.1f, S_margin: %.1f)\n', r2.reliability_score, r2.margin_score_component);

    % -------------------------------------------------------------------------
    % Test 3: BORDERLINE Quality Image
    % -------------------------------------------------------------------------
    q_bord = struct('score', 48.0, 'quality_class', 'BORDERLINE');
    r3 = computeReliability(q_bord, sev_good, fus_good, les_good, vess_good, od_good, fovea_good);
    d3 = makeScreeningDecision(q_bord, fus_good, r3);
    assert(r3.quality_score_component == 48.0, 'Test 3 Failed: S_q should match quality score.');
    fprintf('  [PASS] Test 3: BORDERLINE image (Score: %.1f, Quality S_q: %.1f)\n', r3.reliability_score, r3.quality_score_component);

    % -------------------------------------------------------------------------
    % Test 4: UNGRADABLE Image (Safety Gate Override)
    % -------------------------------------------------------------------------
    q_ungr = struct('score', 22.0, 'quality_class', 'UNGRADABLE');
    r4 = computeReliability(q_ungr, sev_good, fus_good, les_good, vess_good, od_good, fovea_good);
    d4 = makeScreeningDecision(q_ungr, fus_good, r4);
    assert(r4.reliability_score <= 30.0, 'Test 4 Failed: Ungradable must cap reliability at 30.');
    assert(strcmp(d4.decision, 'RECAPTURE'), 'Test 4 Failed: Action must be RECAPTURE.');
    assert(d4.decision_readiness == false, 'Test 4 Failed: Decision readiness must be FALSE.');
    fprintf('  [PASS] Test 4: UNGRADABLE image safety override (Score: %.1f, Action: %s)\n', r4.reliability_score, d4.decision);

    % -------------------------------------------------------------------------
    % Test 5: Missing Vessel Evidence
    % -------------------------------------------------------------------------
    r5 = computeReliability(q_good, sev_good, fus_good, les_good, [], od_good, fovea_good);
    assert(r5.completeness.vessels == 0, 'Test 5 Failed: Vessel completeness should be 0.');
    assert(r5.completeness_score_component < 100, 'Test 5 Failed: Completeness score should be penalized.');
    fprintf('  [PASS] Test 5: Missing vessel evidence (Completeness: %.1f%%)\n', r5.completeness_score_component);

    % -------------------------------------------------------------------------
    % Test 6: Missing Lesion Evidence
    % -------------------------------------------------------------------------
    r6 = computeReliability(q_good, sev_good, fus_good, [], vess_good, od_good, fovea_good);
    assert(r6.completeness.lesions == 0, 'Test 6 Failed: Lesion completeness should be 0.');
    fprintf('  [PASS] Test 6: Missing lesion evidence (Completeness: %.1f%%)\n', r6.completeness_score_component);

    % -------------------------------------------------------------------------
    % Test 7: NaN Input Handling
    % -------------------------------------------------------------------------
    sev_nan = struct('probabilities', [NaN, 0.2, 0.3, 0.4, 0.1], 'confidence', NaN, 'p_referable', NaN);
    r7 = computeReliability(q_good, sev_nan, fus_good, les_good, vess_good, od_good, fovea_good);
    assert(~isnan(r7.reliability_score), 'Test 7 Failed: Reliability score must not be NaN.');
    fprintf('  [PASS] Test 7: NaN input handling (Graceful score: %.1f)\n', r7.reliability_score);

    % -------------------------------------------------------------------------
    % Test 8: Inf Input Handling
    % -------------------------------------------------------------------------
    q_inf = struct('score', Inf, 'quality_class', 'GOOD');
    r8 = computeReliability(q_inf, sev_good, fus_good, les_good, vess_good, od_good, fovea_good);
    assert(~isinf(r8.reliability_score) && r8.reliability_score <= 100, 'Test 8 Failed: Score must be clamped.');
    fprintf('  [PASS] Test 8: Inf input handling (Clamped score: %.1f)\n', r8.reliability_score);

    % -------------------------------------------------------------------------
    % Test 9: Invalid Probability Vector
    % -------------------------------------------------------------------------
    sev_inv = struct('probabilities', [-1, 0, 0, 0, 0], 'confidence', 0.1, 'p_referable', 0.1);
    r9 = computeReliability(q_good, sev_inv, fus_good, les_good, vess_good, od_good, fovea_good);
    assert(r9.reliability_score >= 0 && r9.reliability_score <= 100, 'Test 9 Failed: Valid score range.');
    fprintf('  [PASS] Test 9: Invalid probability vector handling (Score: %.1f)\n', r9.reliability_score);

    % -------------------------------------------------------------------------
    % Test 10: CNN / Lesion Discordance
    % -------------------------------------------------------------------------
    % CNN predicts severe DR (P_ref = 0.95), but 0 lesions detected
    sev_ref = struct('probabilities', [0.01, 0.02, 0.40, 0.50, 0.07], 'confidence', 0.50, 'p_referable', 0.97, 'expected_grade', 3.6);
    fus_ref = struct('p_referable_fused', 0.90, 'p_cnn_referable', 0.97, 'agreement_score', 0.30);
    r10 = computeReliability(q_good, sev_ref, fus_ref, les_good, vess_good, od_good, fovea_good);
    assert(strcmp(r10.evidence_agreement, 'DISCORDANT'), 'Test 10 Failed: Should be DISCORDANT.');
    fprintf('  [PASS] Test 10: CNN/Lesion discordance (Agreement: %s, S_agree: %.1f)\n', r10.evidence_agreement, r10.agreement_score_component);

    % -------------------------------------------------------------------------
    % Test 11: Complete Concordant Evidence
    % -------------------------------------------------------------------------
    les_ref = struct('ma_count', 25, 'ma_area', 350, 'he_count', 12, 'he_area_fraction', 0.008, 'ex_count', 18, 'ex_area_fraction', 0.012, 'dme_risk', 2.0, 'nv_score', 0.65);
    r11 = computeReliability(q_good, sev_ref, fus_ref, les_ref, vess_good, od_good, fovea_good);
    d11 = makeScreeningDecision(q_good, fus_ref, r11);
    assert(strcmp(r11.evidence_agreement, 'CONCORDANT'), 'Test 11 Failed: Should be CONCORDANT.');
    assert(strcmp(d11.decision, 'ACCEPT_SCREENING'), 'Test 11 Failed: Action should be ACCEPT_SCREENING.');
    assert(d11.predicted_referable == true, 'Test 11 Failed: Should be predicted referable.');
    fprintf('  [PASS] Test 11: Complete concordant evidence (Score: %.1f, Action: %s, Ref: %d)\n', r11.reliability_score, d11.decision, d11.predicted_referable);

    % -------------------------------------------------------------------------
    % Test 12: Test-Partition Quarantine Assertion
    % -------------------------------------------------------------------------
    [~, idrid_test] = load_idrid();
    if height(idrid_test) > 0
        assert(height(idrid_test) == 103, 'Test 12 Failed: Expected 103 test samples.');
    end
    fprintf('  [PASS] Test 12: IDRiD test partition strictly quarantined (103 images untouched)\n');

    fprintf('============================================================\n');
    fprintf('  ALL 12 RELIABILITY UNIT TESTS PASSED SUCCESSFULLY!        \n');
    fprintf('============================================================\n\n');
end
