---
marp: true
theme: default
paginate: true
size: 16:9
math: katex
style: |
  :root {
    --ink: #132c46;
    --muted: #5b6f85;
    --blue: #1d62d0;
    --cyan: #0b9fc7;
    --green: #15965d;
    --orange: #d97706;
    --red: #c24133;
    --bg: #f4f8fd;
    --panel: #ffffff;
    --line: #d8e5f2;
    --soft: #eaf3ff;
  }

  section {
    font-family: Inter, Aptos, Arial, sans-serif;
    background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%);
    color: var(--ink);
    padding: 42px 54px 36px 54px;
  }

  h1 {
    font-size: 50px;
    line-height: 1.03;
    letter-spacing: -1.5px;
    margin: 0 0 20px 0;
    color: var(--ink);
    font-weight: 900;
  }

  h2 {
    font-size: 34px;
    line-height: 1.08;
    letter-spacing: -0.8px;
    margin: 0 0 18px 0;
    color: var(--ink);
    font-weight: 850;
  }

  h3 {
    font-size: 23px;
    line-height: 1.12;
    margin: 0 0 10px 0;
    font-weight: 850;
    color: var(--ink);
  }

  p, li {
    font-size: 21px;
    line-height: 1.34;
  }

  strong {
    color: var(--ink);
    font-weight: 900;
  }

  .kicker {
    text-transform: uppercase;
    letter-spacing: 1.8px;
    font-size: 14px;
    font-weight: 900;
    color: var(--cyan);
    margin-bottom: 10px;
  }

  .lead {
    font-size: 28px;
    line-height: 1.22;
    font-weight: 700;
    max-width: 1050px;
  }

  .muted {
    color: var(--muted);
  }

  .small {
    font-size: 15px;
    line-height: 1.28;
  }

  .tiny {
    font-size: 12px;
    line-height: 1.18;
  }

  .chapter {
    background: linear-gradient(135deg, #112b46 0%, #165db4 100%);
    color: white;
  }

  .chapter h1, .chapter h2, .chapter p, .chapter .kicker, .chapter .muted {
    color: white;
  }

  .quote {
    border-left: 8px solid var(--cyan);
    background: rgba(255, 255, 255, 0.78);
    border-radius: 0 20px 20px 0;
    padding: 18px 22px;
    font-size: 28px;
    line-height: 1.25;
    font-weight: 850;
  }

  .chapter .quote {
    background: rgba(255, 255, 255, 0.13);
    border-left-color: #8be4ff;
    color: white;
  }

  .grid2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: stretch;
  }

  .grid3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 18px;
    align-items: stretch;
  }

  .grid4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    align-items: stretch;
  }

  .card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 20px 22px;
    box-shadow: 0 14px 34px rgba(22, 50, 79, 0.08);
  }

  .card.compact {
    padding: 16px 18px;
  }

  .tag {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 900;
    margin: 0 6px 8px 0;
    background: #e8f3ff;
    color: #135baf;
  }

  .tag.green { background: #e7f8ef; color: #0e7245; }
  .tag.orange { background: #fff4df; color: #9a5700; }
  .tag.red { background: #ffeceb; color: #a83226; }
  .tag.gray { background: #eef3f8; color: #4b5e70; }

  .metric {
    font-size: 42px;
    line-height: 0.95;
    font-weight: 950;
    color: var(--blue);
  }

  .metric-label {
    font-size: 14px;
    font-weight: 850;
    color: var(--muted);
    margin-top: 6px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    background: white;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 12px 28px rgba(22, 50, 79, 0.08);
    font-size: 13px;
  }

  th {
    background: #1f4e79;
    color: white;
    font-weight: 850;
    padding: 8px 8px;
    text-align: left;
    vertical-align: top;
  }

  td {
    padding: 7px 8px;
    border-bottom: 1px solid #e4edf6;
    vertical-align: top;
  }

  .bigtable table { font-size: 11.5px; }
  .bigtable th { padding: 7px 6px; }
  .bigtable td { padding: 6px 6px; }

  img {
    max-width: 100%;
    border-radius: 16px;
  }

  .img-card {
    background: white;
    border-radius: 22px;
    padding: 12px;
    box-shadow: 0 14px 34px rgba(22, 50, 79, 0.09);
    border: 1px solid var(--line);
  }

  .row {
    display: flex;
    align-items: center;
    gap: 18px;
  }

  .pillline {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
  }

  .flow {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-top: 26px;
  }

  .flow .node {
    background: white;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 16px 14px;
    min-height: 110px;
    box-shadow: 0 12px 30px rgba(22, 50, 79, 0.08);
  }

  .node .num {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: var(--blue);
    color: white;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    margin-bottom: 8px;
  }

  .answer {
    background: #edf7ff;
    border: 1px solid #c9e5ff;
    border-radius: 18px;
    padding: 16px 18px;
    margin-top: 16px;
    font-size: 18px;
    line-height: 1.28;
    font-weight: 700;
  }

  .warning {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 18px;
    padding: 15px 18px;
    font-size: 18px;
    line-height: 1.28;
  }

  .placeholder {
    min-height: 250px;
    border: 2px dashed #b9cbe0;
    background: rgba(255, 255, 255, 0.56);
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: var(--muted);
    font-size: 15px;
    font-weight: 800;
    padding: 18px;
  }

  .dense li {
    font-size: 16px;
    line-height: 1.22;
    margin-bottom: 5px;
  }

  footer {
    color: #7b8ca0;
    font-size: 10px;
  }
---

<!-- _class: chapter -->

<div class="kicker">Soutenance projet industriel</div>

# Détection d'anomalies industrielles

<div class="lead">De l'image brute à une décision qualité : détecter une pièce défectueuse et localiser la zone suspecte.</div>

<div class="quote" style="margin-top: 34px;">Comprendre les données, préserver le signal utile, puis adapter le modèle au contexte industriel.</div>

---

# Objectifs du projet

<div class="grid2">
<div class="card">
<h2>Besoin qualité</h2>
<p>Décider si une pièce est normale ou défectueuse, avec une heatmap exploitable par un inspecteur.</p>
<div class="pillline">
<span class="tag">détection image</span>
<span class="tag green">localisation pixel</span>
<span class="tag orange">faible faux positif</span>
<span class="tag violet">limiter les non conformités</span>
</div>
</div>

<div class="card">
<h2>Livrables data science</h2>
<p>EDA, harmonisation, preprocessing, baselines, modèles avancés, métriques, interprétation et démonstrateur Streamlit.</p>
<div class="pillline">
<span class="tag gray">notebooks</span>
<span class="tag gray">GitHub</span>
<span class="tag gray">rapport</span>
<span class="tag gray">démo</span>
</div>
</div>
</div>

<div class="answer">Le score global ne suffit pas : une bonne classification image reste insuffisante si la localisation n'est pas utilisable.</div>

---

# Notre boucle expérimentale

<div class="flow" style="margin-top: 8px;">
<div class="node"><div class="num">1</div><h3>Tester</h3><p class="small">Modèle simple ou variante ciblée</p></div>
<div class="node"><div class="num">2</div><h3>Évaluer</h3><p class="small">AUROC, AUPIMO, heatmaps, erreurs</p></div>
<div class="node"><div class="num">3</div><h3>Diagnostiquer</h3><p class="small">Modèle, preprocessing, ROI ou seuil</p></div>
<div class="node"><div class="num">4</div><h3>Itérer</h3><p class="small">Crops, ROI, layers, post-traitement</p></div>
<div class="node"><div class="num">5</div><h3>Décider</h3><p class="small">Conserver, abandonner ou spécialiser</p></div>
</div>

<div class="grid3" style="margin-top: 22px;">
<div class="card compact"><h3>Image AUROC / AP</h3><p class="small">Décision pièce normale ou défectueuse.</p></div>
<div class="card compact"><h3>Pixel AUROC / AP</h3><p class="small">Qualité de localisation de la heatmap.</p></div>
<div class="card compact"><h3>AUPIMO</h3><p class="small">Localisation exploitable avec peu de faux positifs.</p></div>
</div>

<div class="quote" style="margin-top: 22px;">Notre métrique métier centrale devient l’AUPIMO : localiser juste, sans transformer l’inspection en chasse aux faux positifs.</div>

---

# Partie 1 · EDA : à quoi ressemblent les datasets

<div class="img-card">

<img src="figures/eda_previews_6_categories_grid_3x2.png" style="max-height: 410px; width: auto;" />

</div>

<div class="answer">Ces exemples montrent pourquoi on ne peut pas traiter screw, toothbrush, bottle, cable, casting_class1 et STEEL avec exactement la même préparation image.</div>

---

# EDA : premiers constats mesurés

<div class="grid3" style="margin-top: 22px;">
<div class="card compact"><h3>MVTec AD</h3><p class="small">Benchmark propre, 15 catégories, masques disponibles, structure standard.</p></div>
<div class="card compact"><h3>HSS IAD</h3><p class="small">Dataset plus difficile, pièces métalliques, défauts subtils, forte variabilité visuelle.</p></div>
<div class="card compact"><h3>Décision</h3><p class="small">Ne pas lire uniquement une moyenne globale. Les résultats doivent être analysés par catégorie.</p></div>
</div>

<div class="grid2" style="margin-top: 18px;">
<div class="img-card">

![](figures/eda_distribution_categories.png)

</div>
<div class="img-card">

![](figures/eda_localisation_spatiale_defauts.png)

</div>
</div>

<div class="answer">17 429 images, 22 catégories, 2 sources : volumes déséquilibrés, résolutions hétérogènes, micro-défauts et défauts parfois hors centre.</div>

---

# Conséquence : pas de preprocessing unique

<div class="grid2">
<div class="card dense">
<h2>Géométrie et ROI</h2>
<ul>
<li>resize contrôlé, éviter le resize agressif 224/256 ;</li>
<li>center_crop seulement quand il ne coupe pas le défaut ;</li>
<li>tiling + stride pour les formats extrêmes ;</li>
<li>variation local / contexte pour garder détail et structure ;</li>
<li>ROI érodée, foreground mask, ROI métier et landmarks.</li>
</ul>
</div>

<div class="card dense">
<h2>Robustesse et apprentissage</h2>
<ul>
<li>denoising et filtrage qualité quand le fond domine ;</li>
<li>simulation de défauts pour certaines familles ;</li>
<li>Masked Modeling et Masked Semantic Inpainting comme pistes ;</li>
<li>augmentations : rotations, flip, color-jitter, normalisation photométrique ;</li>
<li>équilibrage et stratification pour évaluer sans moyenne trompeuse.</li>
</ul>
</div>
</div>

<div class="quote" style="margin-top: 22px;">Le preprocessing définit ce que le modèle a le droit de voir et ce que le score doit ignorer.</div>

---

# Acte 2 · Baselines : diagnostiquer avant d'optimiser

<div class="bigtable">

| Baseline | Paramètres principaux | Hypothèse | Constat | Décision |
|---|---|---|---|---|
| Autoencodeur | ConvAE, batch 1, 5 epochs, reconstruction image | Une anomalie doit être mal reconstruite | Signal trop diffus, faible pouvoir discriminant | Baseline historique uniquement |
| PatchCore | ResNet18 gelé, layers 2 et 3, tuiles 384, overlap 0.50, kNN euclidien | Une zone anormale est éloignée des patches normaux | Plus robuste et plus local | Baseline forte et interprétable |

</div>

<div class="quote" style="margin-top: 22px;">Première leçon : le modèle diagnostique une limite, mais il ne peut pas retrouver un défaut détruit par le preprocessing.</div>

---

# Dinomaly change la donne

<div class="grid2">
<div class="card">
<h2>Pourquoi ça change</h2>
<p>Dinomaly utilise DINOv2 et reconstruit des features normales plutôt que des pixels. Les défauts topologiques et structurels deviennent plus lisibles.</p>
</div>
<div class="card">
<h2>Ce qui reste à adapter</h2>
<p>La performance dépend encore du périmètre, de la résolution, des crops, des ROI, du post-traitement et du checkpoint retenu.</p>
</div>
</div>

<div class="quote" style="margin-top: 24px;">Dinomaly améliore nettement MVTec et cable, mais ne résout pas automatiquement STEEL ni Casting.</div>

---

# MVTec : validation benchmark

<div class="grid2">
<div>
<p>Avant de conclure sur les cas industriels, on vérifie que le pipeline fonctionne sur un standard reconnu.</p>
<ul>
<li>15 catégories MVTec AD ;</li>
<li>comparaison Dinomaly, PatchCore et approches complémentaires ;</li>
<li>normalisation globale par modèle ;</li>
<li>ensemble Mean : Dinomaly + PatchCore manuel.</li>
</ul>
</div>
<div class="card">
<div class="metric">0.843</div>
<div class="metric-label">AUPIMO moyen avec Ensemble Mean</div>
<p class="small" style="margin-top: 18px;">Dinomaly seul : 0.763. Référence moderne : environ 0.86.</p>
</div>
</div>

<div class="quote" style="margin-top: 20px;">Réussir MVTec ne suffit pas, mais cela prouve que la chaîne est techniquement crédible.</div>

---

# Cable : notre cas de réussite vérifié

<div class="grid2">
<div>

| Scénario | Image AUROC | Pixel AUROC | AUPIMO |
|---|---:|---:|---:|
| PatchCore V7 | 0.762 | 0.847 | 0.075 |
| Dinomaly V11 | 0.960 | 0.922 | 0.380 |
| Dinomaly V13 best | 0.957 | 0.927 | 0.443 |
| Dinomaly final 3cat | 0.991 | 0.923 | 0.576 |
| Dinomaly final MVTec | 0.998 | 0.976 | 0.783 |

</div>
<div class="img-card">

![](figures/preview_cable.png)

</div>
</div>

<div class="answer">Cable montre que le triptyque ROI, preprocessing par catégorie et Dinomaly peut produire une localisation beaucoup plus exploitable que la baseline PatchCore.</div>

---

# STEEL : Dinomaly ne suffit pas

<div class="grid2">
<div class="card dense">
<h2>Difficulté</h2>
<ul>
<li>résolution native 256 x 1600, ratio 1:6.25 ;</li>
<li>forte variabilité intra-normal : surfaces lisses, striées, masquées ;</li>
<li>défauts subtils, localisés et peu contrastés ;</li>
<li>les heatmaps peuvent scorer le bruit sain au lieu du défaut.</li>
</ul>
</div>
<div>

| Étape | Score clé | Lecture |
|---|---:|---|
| ConvAE | AUROC image 0.531 | reconstruction pixel insuffisante |
| DINOv2 / PatchCore | AUROC image 0.572 | transfert limité |
| SuperSimpleNet | AUROC pixel 0.760 | meilleure localisation |
| `tail_mean` | AUROC image 0.638 | meilleure agrégation pixel -> image |

</div>
</div>

<div class="quote" style="margin-top: 22px;">STEEL est un échec informatif : la difficulté vient autant de l'agrégation et de la variabilité saine que du choix du backbone.</div>

---

# Casting : ROI métier + Reverse Distillation calibré

<div class="grid2">
<div class="card dense">
<h2>Pourquoi c'est difficile</h2>
<ul>
<li>micro-défauts, trous, bords et filetages proches des anomalies ;</li>
<li>plusieurs patterns de vue dans Casting_class1 ;</li>
<li>le resizing global dilue les défauts ;</li>
<li>avant de détecter, il faut apprendre où regarder.</li>
</ul>
</div>
<div class="card dense">
<h2>Réponse retenue</h2>
<ul>
<li>ROI de surface fonctionnelle pour isoler la zone inspectable ;</li>
<li>séparation ROI d'affichage / ROI de scoring ;</li>
<li>Reverse Distillation sur features normales ;</li>
<li>calibration post-hoc : layer2 0.65, layer3 0.35, top-k 0.005.</li>
</ul>
</div>
</div>

<div class="metric" style="font-size: 30px; margin-top: 18px;">image AP 0.8461 · pixel AP 0.2251 · AUPIMO 0.0571</div>

---

# Lecture transversale des trois classes

<div class="bigtable">

| Classe | Preprocessing clé | Modélisation | Résultat / limite |
|---|---|---|---|
| Cable | ROI, érosion, crops adaptés | Dinomaly | réussite nette, AUPIMO final 3cat 0.576 |
| STEEL | tuilage, masque steel, agrégation robuste | SuperSimpleNet + `tail_mean` | pixel AUROC 0.760, image AUROC 0.638 |
| Casting | ROI métier, landmarks, scoring local | Reverse Distillation calibré | image AP fort, localisation encore limitée |

</div>

<div class="quote" style="margin-top: 22px;">La conclusion scientifique n'est pas “un meilleur modèle pour tout”, mais “une stratégie par catégorie selon géométrie, défauts et usage qualité”.</div>

---

# Ce que nous avons réellement appris

<div class="grid2">
<div class="card">
<h2>Le modèle compte</h2>
<p>PatchCore, PaDiM et Dinomaly ne font pas le même compromis entre texture locale, structure globale et reconstruction de features.</p>
</div>
<div class="card">
<h2>Mais la chaîne compte plus</h2>
<p>ROI, crops, résolution utile, seuils, post traitement et choix du checkpoint changent directement la qualité des heatmaps.</p>
</div>
</div>

<div class="quote" style="margin-top: 28px;">La performance utile vient de l’alignement entre données, preprocessing, métrique et usage qualité.</div>

---

<!-- _class: chapter -->

<div class="kicker">Conclusion</div>

# Notre épopée en une phrase

<div class="lead">Nous sommes partis d’une détection générique d’anomalies, et nous avons construit progressivement une chaîne plus métier : regarder les bonnes zones, comparer les modèles avec les bonnes métriques, puis stabiliser la décision.</div>

<div class="quote" style="margin-top: 34px; max-width: 1050px;">Pour détecter correctement un défaut industriel, il ne suffit pas d’avoir un bon modèle : il faut préserver le signal utile et apprendre au pipeline où regarder.</div>

<div class="small" style="margin-top: 34px; opacity: 0.9;">Transition démo : image brute, preprocessing / ROI, modèle, heatmap, score image, masque de référence, décision qualité.</div>
