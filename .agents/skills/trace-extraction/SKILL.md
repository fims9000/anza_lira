---
name: trace-extraction
description: Build model-native axial geometry, skeleton graphs, trace segments, confidence, metrics, and GeoJSON.
---

# Trace Extraction

Use master-spec sections 13–21. Treat orientation as axial (`theta == theta +
pi`) using doubled angles. Test orientation periodicity, skeleton graph,
junctions, GeoJSON, and trace metrics first. Threshold masks only with the frozen
validation threshold. Produce 8-connected skeleton graphs and chains between
endpoints/junctions. Tune merge parameters only on val and export deterministic
LineString features with trace provenance.
