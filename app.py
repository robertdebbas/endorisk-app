"""
EndoRisk : prototype « estimation du risque de fracture ».

Lance :  python3 -m streamlit run app.py
L'interface n'appelle que endorisk_model.load_model() / predict().
"""
import pandas as pd
import streamlit as st
import endorisk_model as em

st.set_page_config(page_title="EndoRisk : prototype", page_icon="🦷", layout="centered")

@st.cache_resource
def get_model():
    return em.load_model()
m = get_model()

KINEMATICS_FR = {"Rotation continue": "Rotary", "Réciprocité": "Reciprocating"}
ALLOY_FR = {"NiTi conventionnel": "Conventional NiTi", "M-Wire": "M-Wire", "Traité thermiquement (CM / Blue)": "Heat-treated (CM/Blue)"}
OPERATOR_FR = {"Endodontiste": "Endodontist", "Résident / étudiant post-gradué": "Graduate student",
               "Étudiant en formation initiale": "Undergraduate student", "Dentiste généraliste": "General practitioner"}
TOOTH_FR = {"Molaire": "Molars", "Prémolaire (études sur dents postérieures)": "Posterior",
            "Non spécifique (études toutes dents confondues)": "All"}
USE_POLICY_FR = {"Single tooth": "usage unique", "Up to 4 teeth": "2 à 4 dents", "More than 4 teeth": "plus de 4 dents"}
FACTOR_FR = {"kinematics": "Cinématique", "alloy2": "Type de NiTi", "operator": "Opérateur", "use_policy": "Réutilisation", "tooth_group2": "Dents"}
LEVEL_FR = {**{v: k for k, v in KINEMATICS_FR.items()}, **{v: k for k, v in ALLOY_FR.items()}, **{v: k for k, v in OPERATOR_FR.items()},
            **{v: k for k, v in TOOTH_FR.items()}, "Other/NR": "Antérieure / non précisé", **USE_POLICY_FR}

st.title("EndoRisk : estimation du risque de fracture")
st.caption("Prototype. Estimation exploratoire issue de la littérature, pas une probabilité individuelle validée.")

with st.expander("Comment utiliser cette page", expanded=True):
    st.markdown("""
1. **Décrivez la lime et le cas** dans la colonne de gauche.
2. **Lisez le chiffre** « Risque de fracture estimé » et sa fourchette. Ils se mettent à jour à chaque changement.
3. **Comparez-le** au taux moyen de la littérature affiché à côté.

**D'où vient ce chiffre ?** Nous avons rassemblé 52 groupes d'instruments décrits dans des études publiées
(environ 76 000 instruments ou dents, 1 643 fractures). Pour chaque groupe on connaît le type de lime, le nombre
d'utilisations autorisées, le type d'opérateur, les dents traitées et le taux de fracture observé. Un modèle
statistique a appris quelles combinaisons vont avec des taux plus hauts ou plus bas, et l'applique au cas décrit.
Les particularités des études (pays, type d'étude) sont moyennées pour que le chiffre reflète la lime et le cas.

**Ce que ce chiffre n'est pas.** Ce n'est pas la probabilité que *votre* lime casse. Le modèle n'a jamais vu un
instrument individuel, ni la courbure du canal, ni le torque, ni l'état réel de la lime. Prenez-le comme un ordre
de grandeur issu de la littérature.
""")

with st.sidebar:
    st.header("Lime et cas")
    kinematics = KINEMATICS_FR[st.radio("Cinématique", list(KINEMATICS_FR), horizontal=True)]
    alloy = ALLOY_FR[st.selectbox("Type de NiTi", list(ALLOY_FR), help="Tous les instruments du modèle sont en NiTi. Ce choix distingue les familles d'alliage NiTi.")]
    uses = st.number_input("Nombre de dents traitées avec cette lime (y compris celle-ci)", min_value=1, max_value=50, value=1, step=1)
    operator = OPERATOR_FR[st.selectbox("Opérateur", list(OPERATOR_FR))]
    tooth = TOOTH_FR[st.selectbox("Dent traitée", list(TOOTH_FR), help="Les études ne rapportent pas les fractures par dent ou par canal. Ces groupes correspondent à leurs critères d'inclusion.")]
    st.markdown("---")
    st.caption(f"Régression logistique · version {m['version']} du {m['trained_on']} · {m['n_arms']} groupes d'étude, {m['n_units']:,} instruments ou dents, {m['n_fractured']:,} fractures")

case = dict(kinematics=kinematics, alloy=alloy, operator=operator, uses_teeth=int(uses), tooth_group=tooth)
p = em.predict(m, case)

st.subheader("Estimation issue de la littérature")
c1, c2 = st.columns(2)
c1.metric("Risque de fracture estimé", f"{p['main_pct']:.1f} %",
          help="Régression logistique ajustée sur des groupes d'instruments issus d'études publiées. "
               "La fourchette est un intervalle à 95 % obtenu en réajustant le modèle sur des tirages au sort des études.")
c1.caption(f"Fourchette (95 %) : **{p['main_lo']:.1f} – {p['main_hi']:.1f} %**")
c2.metric("Taux moyen dans la littérature", f"{p['base_rate_pct']:.1f} %", help="Toutes études confondues, pour comparaison")
st.caption(f"Réutilisation retenue : **{USE_POLICY_FR[p['levels']['use_policy']]}**.")

st.warning("Estimation exploratoire basée sur des données agrégées de la littérature (groupes d'instruments issus d'études). "
           "Ce n'est pas une probabilité individuelle validée : le modèle n'a pas encore été entraîné ni évalué sur des cas cliniques réels.")

with st.expander("Repères littérature pour chaque facteur choisi"):
    ref = pd.DataFrame(p["references"])
    ref["Facteur"] = ref["factor"].map(FACTOR_FR); ref["Niveau"] = ref["level"].map(LEVEL_FR).fillna(ref["level"])
    ref = ref.rename(columns={"arms": "Groupes d'étude", "n": "Unités", "fractured": "Fractures", "rate_pct": "Taux %"})
    ref["Taux %"] = ref["Taux %"].round(2)
    st.dataframe(ref[["Facteur", "Niveau", "Groupes d'étude", "Unités", "Fractures", "Taux %"]], hide_index=True, width="stretch")
    st.bar_chart(ref.set_index("Facteur")["Taux %"], horizontal=True)
    st.caption("Taux brut poolé par niveau de facteur, sans ajustement. Un niveau avec 1 ou 2 groupes d'étude n'est pas fiable.")

with st.expander("Détails du modèle"):
    st.markdown(f"""
- **D'où viennent les données.** {m['n_arms']} groupes d'instruments décrits dans des études cliniques publiées,
  rassemblés principalement à partir de la revue systématique de Gomes et collaborateurs (2021). Au total
  {m['n_units']:,} instruments ou dents et {m['n_fractured']:,} fractures, soit un taux moyen de {m['base_rate_pct']:.2f} %.
- **Quel modèle.** Une régression logistique, c'est-à-dire un modèle statistique classique qui relie le taux de fracture
  d'un groupe à ses caractéristiques. Les groupes comptant plus d'instruments pèsent plus lourd. Le modèle est volontairement
  « freiné » (pénalisé) pour ne pas tirer de conclusions trop fortes de quelques études : avec une cinquantaine de groupes,
  un modèle sans frein exagérait les écarts entre cas.
- **Quelles variables.** Celles que vous saisissez : cinématique, type de NiTi, réutilisation, opérateur, dents. Plus des
  variables décrivant les études elles-mêmes : pays, époque, type d'étude, unité de comptage. Ces dernières servent à corriger
  les comparaisons, pas à décrire votre cas : pays, type d'étude et unité sont moyennés sur l'ensemble des études, et l'époque
  est fixée à la période la plus récente.
- **La fourchette.** On a réajusté le modèle {m['ridge_boot'].shape[0]} fois en tirant au sort les études incluses, et on
  retient l'intervalle qui contient 95 % des résultats. Elle dit à quel point les études se contredisent pour ce type de cas.
  Elle ne dit rien de l'incertitude pour un instrument précis.
- **Comment il a été vérifié.** Chaque étude a été mise de côté à tour de rôle, le modèle réappris sans elle, puis utilisé pour
  prédire son taux de fracture. Sur ce test, le modèle fait mieux qu'une simple moyenne, de façon modeste mais réelle, et ses
  prédictions sont bien calibrées : quand il annonce 3 %, on observe environ 3 %. Il explique environ un quart des différences
  de taux entre études ; les trois quarts restants tiennent à des facteurs que la littérature ne décrit pas.
- **Ce qui manque.** Des cas cliniques individuels, avec la courbure du canal, les réglages du moteur et l'historique réel de
  la lime. C'est l'objet de la collecte en cours à l'USJ. Le modèle sera remplacé quand ces données existeront.
""")
    if st.checkbox("Afficher le modèle de contrôle (forêt aléatoire recalibrée)", value=False):
        st.markdown(f"Un second modèle, de type forêt aléatoire, donne pour ce cas **{p['control']['pct']:.1f} %**. "
                    "Lors de la vérification, il faisait jeu égal avec la régression ; il sert uniquement de contrôle de cohérence.")
