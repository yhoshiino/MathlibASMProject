# Module Moteur physique — Optimisation vectorielle et matricielle (C++20, SSE/SSE2, ASM x64)

Rapport individuel du premier module du projet fil rouge.
Bibliothèque mathématique réutilisée (Vecteurs / Matrices), complétée par trois traitements par lots optimisés, une version SoA, une fonction ASM x64, une application console de benchmark et des vérifications automatiques.

> Les passages marqués **À COMPLÉTER** sont des éléments que je n'ai pas encore produits ou dont je dois recopier la valeur exacte depuis mon poste.

---

## 1. Organisation de la solution

| Projet / fichier | Rôle |
|---|---|
| `MathLibrary/Vector3.h`, `Matrice4x4.h` | Référence C++ scalaire (`math::scalar`, templates, `float`) |
| `MathLibrary/Vector3SIMD.h`, `Matrix4x4SIMD.h` | Versions SSE2 (`math::simd`) |
| `MathLibrary/BatchOps.h/.cpp` | Traitements par lots : `dot`, `normalize`, `transform` en versions ref, ref sans autovectorisation, SSE (AoS), SSE (SoA) + conversions AoS ↔ SoA |
| `MathLibrary/DotVec3.asm` | Fonctions ASM x64 (MASM) : `dot_vec3_asm`, `dot_batch_asm` |
| `Benchmark/main.cpp` | Application console : vérifications, mesures nanobench, affichage |
| Projet de tests unitaires | **À COMPLÉTER** (projet séparé exigé par le cahier des charges) |

L'API de la bibliothèque est indépendante du rendu et du gameplay. Aucune bibliothèque externe ne remplace les calculs optimisés (nanobench ne sert qu'au chronométrage).

---

## 2. Conventions documentées

- **Stockage des matrices** : column-major, `data[col*4 + row]`, colonne `j` = `col[j]` (un registre SSE par colonne).
- **Vecteurs** : vecteurs colonnes, `M * v`. Translation dans la colonne 3. Angles en radians, repère main droite.
- **Composition** : `A * B` applique **B en premier**, puis A. Vérifié par test : `T*R` appliquée à `(1,0,0)` donne `(1,3,3)` ; `R*T` donne `(-2,2,3)`.
- **`math::simd::Vector3`** : 16 octets, `alignas(16)`, composantes `x y z w` ; `w` n'est qu'un padding, toujours 0 en sortie, jamais lu par les calculs. `math::scalar::Vector3` fait 12 octets, sans padding. (Note : la struct anonyme dans une union est une extension MSVC.)
- **Vecteur nul** (`normalize`) : si `|v|² <= 1e-12` (`kNormalizeEpsSq`, soit `|v| < 1e-6`) **ou si `|v|²` est NaN**, le résultat est `(0,0,0)`. La même constante est utilisée par toutes les versions.
- **Transformation de points** : matrice affine (dernière ligne `0 0 0 1`), point avec `w = 1` implicite, **sans division perspective**. Résultat avec `w = 0`.
- **Contrats communs des traitements par lots** :
  - `n = 0` autorisé (rien n'est lu ni écrit), aucune contrainte de multiple de 4 ;
  - aucune lecture ni écriture hors de `[0, n)` ;
  - les `Vector3*` doivent être alignés sur 16 octets (garanti par le type) ; le `float*` de sortie de `dot` n'a aucune contrainte (store non aligné) ;
  - les versions SoA utilisent `loadu`/`storeu` : aucune contrainte d'alignement ;
  - les tableaux d'entrée et de sortie ne doivent pas se chevaucher partiellement.

---

## 3. Implémentations

| Traitement | Réf. C++ | Réf. sans autovec | SSE AoS | SSE SoA | ASM |
|---|---|---|---|---|---|
| Produits scalaires `a[i]·b[i]` | `dot_ref` | `dot_ref_novec` | `dot_sse` | `dot_soa_sse` | `dot_batch_asm` (+ `dot_vec3_asm`) |
| Normalisation | `normalize_ref` | `normalize_ref_novec` | `normalize_sse` | `normalize_soa_sse` | — |
| Transformation affine de points | `transform_ref` | `transform_ref_novec` | `transform_sse` | `transform_soa_sse` | — |

Les versions `*_ref_novec` ont le même corps que `*_ref` avec `#pragma loop(no_vector)` ; elles sont présentées séparément de la référence optimisée.

### 3.1 Ce que contiennent les lanes : AoS contre SoA

**AoS** (tableau de structures) : `x0 y0 z0 w0 | x1 y1 z1 w1 | …`
Un registre SSE contient **les 4 composantes d'un seul vecteur**. Pour un `dot`, `dot_sse` charge 4 vecteurs, fait 4 `mulps`, puis une **transposition 4×4** (`_MM_TRANSPOSE4_PS`) pour obtenir `xs`, `ys`, `zs` ; la lane `w` est inutile. Pour `normalize` et `transform`, un registre = un vecteur, donc un vecteur par itération (pas de reste à gérer).

**SoA** (structure de tableaux) : trois tableaux `xs[]`, `ys[]`, `zs[]`.
Un registre contient **la même composante de 4 vecteurs différents**. Le `dot` devient `(xs·xs' + ys·ys') + zs·zs'` : 3 `mulps` et 2 `addps` pour 4 vecteurs, sans shuffle, toutes les lanes utiles. `normalize` fait 1 `sqrtps` et 3 `divps` pour 4 vecteurs, contre 1 `sqrtps` et 1 `divps` **par vecteur** en AoS.

**Conversions** : `aos_to_soa` et `soa_to_aos` utilisent une transposition 4×4 (SSE). Leur coût est **mesuré séparément** puis **additionné** dans une ligne « total ».

### 3.2 Fonction ASM x64

`DotVec3.asm` (MASM, convention Windows x64 : arguments `rcx`, `rdx`, `r8`, `r9` ; résultat float dans `xmm0`) :

- `dot_vec3_asm(const float* a, const float* b)` : produit scalaire d'un couple de Vec3 ;
- `dot_batch_asm(a, b, out, n)` : même calcul par lots, un vecteur par itération, `n = 0` géré.

L'ordre d'addition est `(x·x' + y·y') + z·z'`, identique à la référence C++ : résultat identique bit à bit.

---

## 4. Choix numériques et techniques justifiés

- **`sqrtps` + `divps` exacts** (pas de `rsqrtps` approximatif) : toutes les versions donnent le même résultat bit à bit, ce qui permet de tester l'égalité stricte. Une normalisation approchée reste une extension possible.
- **Ordre d'addition fixé** `(x+y)+z` partout : les flottants ne sont pas associatifs ; le fixer rend les versions comparables.
- **`/fp:precise`** : règles flottantes identiques pour toutes les versions mesurées.
- **Seuil du vecteur nul à `1e-12` sur `|v|²`** : évite une division par une quantité quasi nulle ; `NaN` est traité comme un vecteur nul grâce à la forme `!(len2 > eps)`.
- **Padding `w` en AoS** : permet de charger un vecteur entier dans un registre avec un alignement de 16 octets. Coût : 25 % de mémoire supplémentaire et une lane gâchée.
- **Tolérances (voir §5.2)** : dérivées du modèle standard de l'arithmétique flottante, pas choisies au hasard.

---

## 5. Validation des calculs

Les vérifications s'exécutent **au lancement de `Benchmark.exe`, avant toute mesure** (les mesures sont annulées en cas d'échec). `Benchmark.exe check` ne lance que les vérifications.

### 5.1 Valeurs connues (calculées à la main)

- `dot` : `(1,2,3)·(4,5,6) = 32`, vecteurs orthogonaux `= 0`, vecteur nul `= 0`, signe négatif ;
- `normalize` : `(3,4,0) → (0.6, 0.8, 0)`, `(0,0,0) → (0,0,0)`, `(0,0,5) → (0,0,1)`, `|v|² < 1e-12 → (0,0,0)`, `NaN → (0,0,0)`, `w = 0` ;
- `transform` : identité (exacte), translation (exacte), rotation Z de 90° `(1,0,0) → (0,1,0)`, composition `T*R` et `R*T` (ordre d'application) ;
- lot vide appelé avec des pointeurs nuls.

### 5.2 Référence en `double` et tolérances justifiées

L'accord entre deux versions ne prouve pas leur correction : chaque résultat est aussi comparé à un calcul en `double` fait à partir des mêmes entrées (4099 vecteurs, aléatoires dans [-10, 10], graine fixe ; `4099 = 4×1024 + 3` exerce le traitement du reste).

Soit `u = 2^-24 ≈ 5,96·10⁻⁸` l'erreur relative maximale d'une opération `float` :

| Traitement | Borne retenue | Justification |
|---|---|---|
| `dot` (3 produits, 2 additions) | `4u · Σ\|x_i·y_i\|` | erreur théorique `γ₃ ≈ 3u`, marge |
| `normalize` | `8u` relatif, par composante | `len²` ≈ 3u, `sqrt` ≈ 1,5u + u, division u → ≈ 3,5u, marge |
| `transform` (4 termes) | `5u · Σ\|termes\|` | erreur théorique `γ₄ ≈ 4u`, marge |

Les tolérances sont **relatives à la somme des termes en valeur absolue**, pas absolues : un résultat proche de zéro obtenu par annulation reste jugé de façon cohérente.

### 5.3 Égalité bit à bit entre versions

`dot` (6 variantes : ref, novec, sse, soa, asm par lots, asm appelé par vecteur), `normalize` (4 variantes), `transform` (4 variantes) sont **identiques bit à bit**. Cette égalité est valable avec `/arch:SSE2` (défaut x64) et `/fp:precise` ; elle n'a pas été testée avec `/arch:AVX2` (une contraction FMA pourrait la casser).

### 5.4 Tailles limites

`n = 0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 1000` : des zones sentinelles de 8 éléments sont placées de part et d'autre du tampon de sortie et vérifiées intactes (**aucune écriture hors `[0, n)`**), et les résultats sont comparés à la référence. L'aller-retour AoS → SoA → AoS est vérifié identique.

### 5.5 Résultats des vérifications (dernière exécution)

| Vérification | Résultat |
|---|---|
| Valeurs connues | OK |
| `dot`, n = 4099 | bit-exact entre 6 versions ; erreur max = **0,55 × la borne** |
| `normalize`, n = 4099 | bit-exact entre 4 versions ; erreur max = **0,30 × la borne** |
| `transform`, n = 4099 | bit-exact entre 4 versions ; erreur max = **0,35 × la borne** |
| Tailles limites | OK |

**Limite connue** : les **lectures** hors bornes ne sont pas détectables avec cette méthode (seules les écritures le sont). **À COMPLÉTER (facultatif)** : exécution avec `/fsanitize=address`.

### 5.6 Tests unitaires

**À COMPLÉTER** : projet de tests séparé (valeurs connues, vecteurs nuls, tailles limites, identité, translation, rotation, composition, comparaison ref/SIMD) à partir des cas de la section 5.1.

---

## 6. Méthodologie de mesure

**Environnement**

| Élément | Valeur |
|---|---|
| CPU | Intel Core Ultra 9 275HX (hybride P-cores + E-cores) |
| Caches (par cache) | L1d 48 Ko, L2 4 Mo, L3 36 Mo |
| Compilateur | MSVC, `_MSC_VER = 1944`, C++20 (`_MSVC_LANG = 202002`) |
| Configuration | Release x64, **sans débogueur** (Ctrl+F5) |
| Jeu d'instructions | SSE2 (défaut x64), intrinsics SSE/SSE2 uniquement |
| Flottants | `/fp:precise` |
| Options de compilation | `/Qvec-report:2` |

**Protocole** (nanobench)

- données reproductibles : générateur nanobench, **graine fixe 42** ;
- données **préparées hors chronométrage** ; aucune allocation ni affichage dans la zone mesurée ;
- **échauffement** de 20 itérations, **21 époques**, au moins 10 itérations par époque et 10 ms par époque ;
- résultats empêchés d'être supprimés par `doNotOptimizeAway` ;
- aucun traitement ne modifie ses entrées : pas de réinitialisation nécessaire entre les essais ;
- thread **épinglé sur un P-core** (cœur logique 0, `EfficiencyClass 1`) pour éviter les migrations entre cœurs hybrides ;
- trois tailles de lots : **petite (n = 16)**, **moyenne (n = 4096)**, **grande (n = 1 048 576)** ;
- métriques : médiane (ns/élément), dispersion (`err%` = erreur absolue médiane en pourcentage), `relative` = temps de la ligne de base / temps de la ligne (100 % = référence, 300 % = 3× plus rapide).

**Ligne de base** : la référence C++ **optimisée, auto-vectorisation autorisée** (`*_ref`). La variante sans autovectorisation est identifiée séparément (`*_ref_novec`).

**Relancer les mesures**

```
Benchmark.exe                       # tout, 3 tailles
Benchmark.exe dot                   # un traitement, toutes versions
Benchmark.exe normalize soa 4096    # un traitement, une version (+ réf), une taille
Benchmark.exe check                 # vérifications seules
```

Traitements : `all | dot | normalize | transform | vector3 | matrix | check`. Versions : `all | novec | sse | soa | asm` (`asm` seulement pour `dot`).

---

## 7. Résultats

Valeurs : **ns/élément** (médiane), et entre parenthèses le `relative` par rapport à la référence C++ optimisée.
Mémoire touchée : `dot` 0,6 Ko / 144 Ko / 36 Mo ; `normalize` et `transform` 0,5 Ko / 128 Ko / 32 Mo (pour n = 16 / 4096 / 1 048 576).

### 7.1 `dot`

| Version | n = 16 | n = 4096 | n = 1 048 576 |
|---|---|---|---|
| ref (AoS, autovec autorisée) | 0,677 (100 %) | 0,732 (100 %) | 1,617 (100 %) |
| ref sans autovec | 0,681 (99 %) | 0,701 (104 %) | 1,676 (96 %) |
| SSE AoS | 0,307 (**220 %**) | 0,353 (**207 %**) | 1,276 (127 %) |
| ASM AoS (1 vecteur/itération) | 0,505 (134 %) | 0,505 (145 %) | 1,816 (89 %) |
| conv AoS→SoA (1 tableau) | 0,292 | 0,488 | 0,953 |
| SoA, calcul seul | 0,353 (192 %) | 0,244 (**300 %**) | 0,835 (194 %) |
| **SoA, total (2 conv + calcul)** | 0,859 (**79 %**) | 1,303 (**56 %**) | 3,507 (**46 %**) |

### 7.2 `normalize`

| Version | n = 16 | n = 4096 | n = 1 048 576 |
|---|---|---|---|
| ref | 2,491 (100 %) | 2,412 (100 %) | 2,654 (100 %) |
| ref sans autovec | 2,420 (103 %) | 2,413 (100 %) | 2,587 (103 %) |
| SSE AoS | 1,449 (172 %) | 1,681 (144 %) | 1,807 (147 %) |
| conv AoS→SoA | 0,287 | 0,503 | 0,840 |
| SoA, calcul seul | 0,618 (**403 %**) | 0,606 (**398 %**) | 0,676 (**393 %**) |
| conv SoA→AoS | 0,285 | 0,510 | 0,954 |
| **SoA, total (conv + calcul + conv)** | 0,983 (253 %) | 1,750 (138 %) | 3,066 (**87 %**) |

### 7.3 `transform`

| Version | n = 16 | n = 4096 | n = 1 048 576 |
|---|---|---|---|
| ref | 1,442 (100 %) | 1,399 (100 %) | 1,913 (100 %) |
| ref sans autovec | 1,435 (101 %) | 1,389 (101 %) | 1,879 (102 %) |
| SSE AoS | 0,787 (183 %) | 0,762 (184 %) | 1,397 (137 %) |
| conv AoS→SoA | 0,283 | 0,501 | 0,902 |
| SoA, calcul seul | 0,763 (189 %) | 0,480 (292 %) | 0,664 (288 %) |
| conv SoA→AoS | 0,287 | 0,505 | 0,973 |
| **SoA, total** | 1,190 (121 %) | 1,629 (**86 %**) | 3,406 (**56 %**) |

### 7.4 Dispersion

`err%` est compris entre 0,7 % et 4,6 % pour n = 16 et n = 4096, et monte jusqu'à 9,1 % à n = 1 048 576 (par exemple `dot asm`, 9,1 % ; `dot` sans autovec, 7,5 %). **Un écart de `relative` inférieur à la dispersion n'est pas significatif** : par exemple `dot asm` à 89 % à 1 M éléments est équivalent à la référence.

### 7.5 Fonctions isolées (complément hors cahier des charges)

Ces mesures chronomètrent **un seul appel** et incluent le surcoût du harnais (index masqué, `doNotOptimizeAway`) : les ratios sont **sous-estimés**.

| Fonction | Scalaire (ns/op) | SIMD (ns/op) | relative |
|---|---|---|---|
| `Vector3::dot` | 1,197 | 0,817 | 147 % (ASM : 1,010 → 119 %) |
| `Vector3::magnitude` | 1,031 | 0,944 | 109 % |
| `Vector3::normalized` | 2,471 | 1,496 | 165 % |
| `Mat4x4::translate` | 1,498 | 1,249 | 120 % |
| `Mat4x4::rotationZ` | 6,833 | 7,186 | **95 %** |
| `Mat4x4::operator*` | 18,752 | 3,565 | **526 %** |

---

## 8. Ce que le compilateur produit : auto-vectorisation

Compilation avec `/Qvec-report:2` (rapport émis à l'édition de liens, car l'optimisation du programme entier est active). Seules les lignes de `BatchOps.cpp` sont analysées ici ; le reste du rapport concerne la STL et nanobench.

| Fonction (ligne) | Rapport | Signification du code |
|---|---|---|
| `dot_ref` (16) | non vectorisée, **1501** | aliasing possible sur des tableaux de structures |
| `dot_ref_novec` (236) | non vectorisée, **1400** | `#pragma loop(no_vector)` demandé : le pragma fonctionne |
| `normalize_ref` (48), `transform_ref` (97) | non vectorisées, **1305** | information de type insuffisante |
| `dot_sse` (26 / 38) | 502 / 1501 | boucle principale à pas 4 (rien à vectoriser dans des intrinsics) / reste |
| `dot_soa_sse` (174), `transform_soa_sse` (221) | **vectorisées** | boucle du reste, sur tableaux de `float` séparés |
| `normalize_soa_sse` (192) | non vectorisée, 1100 | le reste contient un `if` |

**Cas bloqué (`dot_ref`, code 1501)** : le pointeur de sortie `out` est un `float*`, et les composantes des `Vector3` sont aussi des `float`. Le compilateur ne peut donc pas exclure qu'écrire `out[i]` modifie `a[i+1].x`, alors que le contrat interdit le chevauchement.

**Cas réussi (boucles de reste en SoA)** : avec des tableaux de `float` distincts et contigus, le compilateur vectorise de lui-même. Le layout SoA aide donc le compilateur autant que les intrinsics.

**Conséquence sur les mesures** : la référence « optimisée » n'est en pratique **pas vectorisée**. Elle produit le même code que la variante `*_ref_novec`, ce qui est cohérent avec des temps identiques (écart ≤ 4 % sur toutes les mesures). Dans le débogueur, la boucle de `dot_ref` est étiquetée `dot_ref_novec` : les deux fonctions ont très probablement été fusionnées par l'éditeur de liens (pliage COMDAT, hypothèse non vérifiée directement).

**Limite** : une variante `dot_ref` déclarée avec `__restrict` n'a pas été testée. **À COMPLÉTER (facultatif)**.

---

## 9. Analyse du code généré (désassemblage)

Extraits obtenus en Release x64 avec le désassemblage du débogueur de Visual Studio.

### 9.1 Référence optimisée — boucle de `dot_ref` (40 instructions pour 4 dots)

Le compilateur déroule la boucle ×4 et utilise uniquement des instructions **scalaires** (`movss`, `mulss`, `addss`) ; la lane `w` n'est jamais lue.

Un vecteur (le premier des quatre) :

```asm
movss  xmm2, [r10+rax-4]   ; charge a.x (r10 = écart entre les deux pointeurs : un seul registre indexe a et b)
mulss  xmm2, [rax-4]       ; xmm2 = a.x * b.x  (la lecture de b est fusionnée dans le mulss)
movss  xmm0, [r10+rax]     ; charge a.y
mulss  xmm0, [rax]         ; xmm0 = a.y * b.y
movss  xmm1, [r10+rax+4]   ; charge a.z
mulss  xmm1, [rax+4]       ; xmm1 = a.z * b.z
addss  xmm2, xmm0          ; x*x' + y*y'
addss  xmm2, xmm1          ; (x*x' + y*y') + z*z'   -> même ordre que la formule C++
movss  [rbx-8], xmm2       ; out[i] = résultat
```

Les trois autres vecteurs reproduisent ce schéma, décalés de 16 octets (taille d'un `Vector3`). Contrôle de boucle, une fois pour 4 vecteurs :

```asm
add  rax, 40h      ; avance de 64 octets = 4 Vector3
add  rbx, 10h      ; avance de 16 octets = 4 floats de sortie
sub  rcx, 1        ; compteur (n/4)
jne  début_de_boucle
```

### 9.2 SIMD explicite — boucle de `dot_sse` (26 instructions pour 4 dots)

```asm
; --- préparation (avant la boucle) ---
sub    r10, rdx             ; r10 = a - b : un seul registre (rax) indexe b ; a = r10 + rax
lea    rax, [rbx+10h]       ; rax = b + 16
; --- boucle ---
movups xmm5,[r10+rax-10h]   ; a[0]  (movups = chargement non aligné)
add    rdx, 4               ; compteur i + 4
mulps  xmm5,[rax-10h]       ; m0 = a[0] * b[0]  (b[0] lu par le mulps : exige 16 octets alignés)
movups xmm4,[r10+rax+10h]   ; a[2]
mulps  xmm4,[rax+10h]       ; m2 = a[2] * b[2]
movups xmm2,[rax+r10]       ; a[1]
movaps xmm3, xmm5           ; copie de m0 (shufps écrase son 1er opérande : SSE à 2 opérandes)
mulps  xmm2,[rax]           ; m1 = a[1] * b[1]
movups xmm0,[r10+rax+20h]   ; a[3]
movaps xmm1, xmm4           ; copie de m2
mulps  xmm0,[rax+20h]       ; m3 = a[3] * b[3]
add    rax, 40h             ; avance de 4 Vector3
; --- transposition 4x4 (_MM_TRANSPOSE4_PS) ---
shufps xmm3,xmm2,44h        ; (x0 y0 x1 y1)
shufps xmm5,xmm2,0EEh       ; (z0 w0 z1 w1)
shufps xmm1,xmm0,44h        ; (x2 y2 x3 y3)
shufps xmm4,xmm0,0EEh       ; (z2 w2 z3 w3)
movaps xmm0, xmm3
shufps xmm0,xmm1,0DDh       ; xmm0 = (y0 y1 y2 y3)  les ys
shufps xmm3,xmm1,88h        ; xmm3 = (x0 x1 x2 x3)  les xs
; --- additions et stockage ---
addps  xmm0, xmm3           ; ys + xs
shufps xmm5,xmm4,88h        ; xmm5 = (z0 z1 z2 z3)  les zs
addps  xmm5, xmm0           ; zs + (ys + xs) = 4 produits scalaires
movups [r8+r11*4], xmm5     ; out[i..i+3]
add    r11, 4               ; i += 4
cmp    rdx, r9              ; i + 4 <= n ?
jbe    début_de_boucle
```

La transposition utilise **7 `shufps`** : le 8ᵉ, qui produirait les `ws`, est supprimé par le compilateur car la lane `w` n'est pas utilisée.

### 9.3 Comparaison pour 4 dots

| | `dot_ref` (scalaire) | `dot_sse` (SIMD) |
|---|---|---|
| Instructions par itération | 40 | **26** |
| Par dot | 10 | 6,5 |
| Lectures mémoire | 24 | 8 |
| Écritures | 4 | 1 |
| Multiplications | 12 (`mulss`) | 4 (`mulps`) |
| Additions | 8 (`addss`) | 2 (`addps`) |
| Shuffles | 0 | 7 |

Le gain mesuré à n = 4096 est de ×2,07 (0,732 → 0,353 ns/élément). Le rapport d'instructions seul (40/26 ≈ ×1,5) ne suffit pas à l'expliquer ; la réduction des lectures mémoire (24 → 8) va dans le sens du gain. Sans compteurs matériels, je présente cette explication comme **cohérente avec** les mesures, pas comme démontrée.

### 9.4 ASM écrit à la main (`dot_batch_asm`)

Boucle de 15 instructions **par dot** (2 `movups`, 1 `mulps`, 2 `shufps`, 2 `addss`, 1 `movss`, 2 `movaps`, 4 instructions de contrôle), contre 10 pour `dot_ref` et 6,5 pour `dot_sse`. Elle est pourtant plus rapide que la référence à n = 4096 (0,505 contre 0,732 ns) : elle effectue 2 lectures mémoire par dot au lieu de 6. Elle reste moins bonne que `dot_sse`, qui traite 4 vecteurs par itération. À 1 M éléments, elle est équivalente à la référence, dans la dispersion.

---

## 10. Analyse des résultats

### 10.1 Gains du SIMD explicite en AoS (`*_sse`)

| Traitement | n = 16 | n = 4096 | n = 1 M |
|---|---|---|---|
| `dot` | ×2,2 | ×2,1 | ×1,27 |
| `normalize` | ×1,7 | ×1,4 | ×1,5 |
| `transform` | ×1,8 | ×1,8 | ×1,4 |

Le SIMD explicite gagne nettement face à une référence non vectorisée aux petites et moyennes tailles. En AoS, un registre ne contient qu'un vecteur (ou 4 vecteurs transposés pour `dot`), avec une lane `w` gâchée : le gain reste limité.

### 10.2 Effet de la taille : la hiérarchie mémoire

À n = 16, les données tiennent dans le L1 ; à n = 4096 (≈ 130–144 Ko), elles dépassent le L1d (48 Ko) mais tiennent dans le L2 (4 Mo) ; à n = 1 048 576, les données (32–36 Mo) atteignent la taille du L3 (36 Mo). Le temps de la référence passe pour `dot` de 0,73 à 1,62 ns/élément, et les gains du SIMD chutent (×2,1 → ×1,27). C'est **cohérent avec** un traitement limité par la bande passante mémoire, où le SIMD ne peut plus accélérer le calcul. Cette explication n'est pas démontrée : aucun compteur matériel n'a été utilisé.

### 10.3 SoA : calcul seul contre coût total

Le **calcul seul** en SoA est le plus rapide : ×3 pour `dot` et `transform` et **×4 pour `normalize`** à n = 4096. Toutes les lanes sont utiles, aucun shuffle n'est nécessaire, et pour `normalize` il faut 1 `sqrtps` et 3 `divps` pour 4 vecteurs au lieu de 1 et 1 **par vecteur**.

Mais **le coût total, conversions comprises, est souvent une régression** :

- `dot` : 79 % / 56 % / 46 % (il faut convertir deux tableaux pour un calcul très court) ;
- `transform` : 121 % / 86 % / 56 % ;
- `normalize` : 253 % / 138 % / 87 % (gain aux petites tailles, grâce au calcul lourd).

Chaque conversion coûte 0,28–0,29 ns/élément à n = 16, 0,49–0,51 à n = 4096 et 0,84–0,97 à 1 M : autant que le calcul AoS lui-même. La somme des trois lignes à n = 4096 pour `normalize` (≈ 1,62) est proche du total mesuré (1,75).

**Conclusion** : le SoA ne vaut le coup que si les données **restent en SoA** pendant plusieurs traitements, ou si le calcul est suffisamment lourd (`normalize`). Convertir juste avant un seul calcul court est un **biais de mesure** classique : mesurer uniquement « SoA, calcul seul » donnerait ×3 à ×4 alors que le coût réel, conversions comprises, est parfois inférieur à la référence.

### 10.4 Limites et cas où le SIMD n'aide pas

- **`rotationZ`** : 95 % (aucun gain). Le temps est dominé par `cos` et `sin`, que le SIMD ne change pas.
- **`operator*`** : ×5,3. Les 4 colonnes sont calculées en parallèle (4 `mulps` × 4 colonnes) au lieu de 64 multiplications scalaires.
- **`magnitude`** : 109 %, gain faible : un seul `sqrt` domine.
- **Grands lots** : le gain s'effondre (voir 10.2).
- **Différences non significatives** : tout écart inférieur à la dispersion (jusqu'à ~9 % à 1 M éléments) n'est pas interprété.

### 10.5 Biais de mesure traités

- Données préparées et allocations hors du chronométrage ; aucun affichage dans la zone mesurée.
- Résultats empêchés d'être supprimés (`doNotOptimizeAway`).
- Thread épinglé sur un P-core pour limiter le bruit des migrations sur CPU hybride.
- Conversions mesurées séparément **et** en coût total.
- Même machine, mêmes entrées, mêmes règles flottantes pour toutes les versions.
- Les fonctions isolées incluent le surcoût du harnais : présentées comme complément, avec ratios sous-estimés.

---

## 11. Profilage CPU

> **À COMPLÉTER** : l'analyse d'au moins un traitement au profileur CPU est exigée par le cahier des charges.
>
> Procédure suggérée : Release x64, arguments `normalize soa 1048576`, *Déboguer > Profileur de performances > Utilisation du processeur*. Dans la vue « Fonctions », trier par **temps CPU propre** et relever la part de `normalize_ref`, `aos_to_soa`, `soa_to_aos` et `normalize_soa_sse`, puis la vue ligne par ligne de `normalize_soa_sse`. Joindre les captures et les pourcentages, et les confronter aux hypothèses de la section 10.3 (poids des conversions face au calcul).

---

## 12. Travail non réalisé et limites

- Profilage CPU et tests unitaires dans un projet séparé : **À COMPLÉTER** (sections 5.6 et 11).
- Aucune extension facultative n'a été traitée (AVX, normalisation approchée, `operator*` par lots, AoSoA, alignement/padding).
- Lectures hors bornes non vérifiées (AddressSanitizer non exécuté).
- Pas de compteurs matériels : les explications microarchitecturales sont des hypothèses cohérentes avec les mesures, pas des preuves.
- Mesures faites sur une seule machine ; non comparables à d'autres CPU sans relancer le benchmark.

---

## 13. Usage de l'IA générative

J'ai utilisé **Claude (Anthropic)** comme assistant pour :

- structurer la bibliothèque `BatchOps` (versions ref, SSE AoS, SSE SoA, conversions AoS ↔ SoA, variantes sans autovectorisation) ;
- écrire la fonction ASM x64 `DotVec3.asm` ;
- écrire l'application de benchmark (`main.cpp`) et les vérifications automatiques (valeurs connues, référence en `double`, égalité bit à bit, tailles limites) ;
- dériver les bornes d'erreur des tolérances ;
- annoter les extraits de désassemblage et lire les codes du rapport `/Qvec-report:2` ;
- rédiger la structure de ce rapport.

Toutes les **mesures proviennent de mes exécutions** sur ma machine ; aucun chiffre de performance n'a été estimé par l'IA.

> **À COMPLÉTER** : ce que j'ai moi-même relu, compris, testé et modifié dans le code fourni par l'IA, et les points où j'ai corrigé ou rejeté ses propositions.
>
> Je dois pouvoir expliquer chaque ligne de `dot_sse` et de `dot_batch_asm` (transposition, restes, alignement) et modifier un extrait pendant la revue de code.
