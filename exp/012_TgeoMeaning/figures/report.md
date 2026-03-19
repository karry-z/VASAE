# Experiment 012: t_geo Meaning Diagnosis — Summary Report

## H3: t_geo is embedding density artifact
  L0: median_margin=0.2444 (null_rot=0.0048, null_rand=0.0048)
  L1: median_margin=0.2264 (null_rot=0.0048, null_rand=0.0049)
  L2: median_margin=0.2203 (null_rot=0.0049, null_rand=0.0049)
  L3: median_margin=0.2020 (null_rot=0.0048, null_rand=0.0049)
  L4: median_margin=0.1869 (null_rot=0.0049, null_rand=0.0049)
  L5: median_margin=0.1651 (null_rot=0.0048, null_rand=0.0048)
  L6: median_margin=0.1482 (null_rot=0.0048, null_rand=0.0049)
  L7: median_margin=0.1376 (null_rot=0.0050, null_rand=0.0049)
  L8: median_margin=0.1483 (null_rot=0.0048, null_rand=0.0049)
  L9: median_margin=0.1701 (null_rot=0.0049, null_rand=0.0049)
  L10: median_margin=0.2575 (null_rot=0.0048, null_rand=0.0049)
  L11: median_margin=0.3046 (null_rot=0.0049, null_rand=0.0048)

## H4: t_geo tokens have biased embedding geometry
  L0: tgeo_norm=3.918 (all=3.959), tgeo_knn=0.538 (all=0.558), coverage=11.1%
  L1: tgeo_norm=3.923 (all=3.959), tgeo_knn=0.537 (all=0.558), coverage=11.1%
  L2: tgeo_norm=3.920 (all=3.959), tgeo_knn=0.538 (all=0.558), coverage=11.1%
  L3: tgeo_norm=3.921 (all=3.959), tgeo_knn=0.537 (all=0.558), coverage=11.1%
  L4: tgeo_norm=3.917 (all=3.959), tgeo_knn=0.538 (all=0.558), coverage=11.1%
  L5: tgeo_norm=3.914 (all=3.959), tgeo_knn=0.537 (all=0.558), coverage=11.0%
  L6: tgeo_norm=3.911 (all=3.959), tgeo_knn=0.538 (all=0.558), coverage=11.0%
  L7: tgeo_norm=3.903 (all=3.959), tgeo_knn=0.538 (all=0.558), coverage=10.9%
  L8: tgeo_norm=3.890 (all=3.959), tgeo_knn=0.540 (all=0.558), coverage=10.9%
  L9: tgeo_norm=3.894 (all=3.959), tgeo_knn=0.539 (all=0.558), coverage=10.9%
  L10: tgeo_norm=3.916 (all=3.959), tgeo_knn=0.538 (all=0.558), coverage=11.0%
  L11: tgeo_norm=3.937 (all=3.959), tgeo_knn=0.537 (all=0.558), coverage=11.1%

## H1: t_geo = mean activation direction
## H2: t_geo = next-token
  L2:
    H1: t_geo==t_mean=1.5%, cos(d,mu)=0.0799
    H2: t_geo==t_next=0.207% (random=0.013%)
  L6:
    H1: t_geo==t_mean=1.1%, cos(d,mu)=0.0918
    H2: t_geo==t_next=0.314% (random=0.023%)
  L11:
    H1: t_geo==t_mean=3.0%, cos(d,mu)=-0.0292
    H2: t_geo==t_next=0.496% (random=0.021%)

## H5: Layer dependence (feature categories)
  L0: alive=767, token_feat=0, context_feat=325, geo=causal=0.5%
  L1: alive=977, token_feat=0, context_feat=382, geo=causal=0.6%
  L2: alive=1128, token_feat=0, context_feat=432, geo=causal=0.1%
  L3: alive=1333, token_feat=0, context_feat=454, geo=causal=0.8%
  L4: alive=1488, token_feat=1, context_feat=530, geo=causal=0.9%
  L5: alive=1563, token_feat=0, context_feat=490, geo=causal=0.8%
  L6: alive=1600, token_feat=0, context_feat=488, geo=causal=1.9%
  L7: alive=1634, token_feat=0, context_feat=510, geo=causal=1.8%
  L8: alive=1471, token_feat=0, context_feat=439, geo=causal=2.8%
  L9: alive=1331, token_feat=0, context_feat=486, geo=causal=1.8%
  L10: alive=1022, token_feat=0, context_feat=522, geo=causal=3.5%
  L11: alive=504, token_feat=1, context_feat=293, geo=causal=10.5%