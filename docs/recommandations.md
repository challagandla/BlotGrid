BlotGrid does not need another detector next. It needs a guided, review-first Quantification workspace.

  The clumsiness is structural:

  - Six sequential tabs force users to manually run Quantify → Normalize → Replicates → Export (app.py:1190).
  - The sidebar mixes routine analysis, model tuning, SAM, training, and project controls (app.py:798).
  - Ordinary users see study-split warnings and training-label approval controls (app.py:1497, app.py:1815).
  - The ROI canvas can send stale full-table snapshots, potentially overwriting a recent table edit (wblot/roi_editor_frontend/
    roi_editor.js:270, wblot/roi_constraints.py:81).

  Recommended sequence:

  1. Stabilize the current ladder/public-source work separately. It currently spans 12 modified and 10 untracked files; keep the root source
     TIFF out of any public commit. The current suite is healthy: 346 passed.

  2. Fix ROI correctness and responsiveness first:
      - Send selected-ROI deltas with revision checks, not full stale snapshots.
      - Avoid reruns for selection-only clicks.
      - Cache the encoded image instead of rebuilding it after every interaction.
      - Add undo, next flagged ROI, and selectable excluded boxes.

  3. Introduce explicit workspaces:
      - Quantify — default and simple.
      - Curate training data — study groups, lineage and annotation approval.
      - Sealed review — automatically locked by evaluator projects.

  4. Replace six tabs with:

     Prepare → Review ROIs → Analyze & Export

     Give each stage one primary action, a readiness checklist, and clear explanations when progression is blocked. Put thresholds, model paths,
     SAM, background algorithms and display geometry under contextual Advanced panels.

  5. Separate scientific states:
      - reviewed for measurement;
      - approved as training annotation;
      - whole-image training attestation.

     These are currently too closely coupled in wblot/project_io.py:60.

  The first implementation slice I would choose is: workspace modes + distinct measurement-review state + one Analyze action. Preserve native-
  pixel densitometry, fixed-size ROI enforcement, provenance checks and sealed-evaluation locks unchanged.

  The Graphify view reinforced this priority: most of the product surface converges on the six-tab workflow and ROI editor, so simplifying that
  path will produce more value than adding another AI capability. No files were changed during this assessment.


