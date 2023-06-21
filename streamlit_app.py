import streamlit as st
from PIL import Image
from bottle_cap_art import generate_image


with st.sidebar:
    st.subheader("Paramètres")
    st.write("Tu peux modifier les paramètres ici")
    count_limit = st.checkbox(
        label="Je veux prendre en compte le nombre de capsules dans le dataset",
        help="Si tu coches cette case, l'image générée n'utiliseras pas plus de capsules que ce que tu as en stock",
        value=False,
    )
    nb_rotations = st.number_input(
        label="Nombre de rotations à essayer pour chaque position de chaque capsule",
        help="Plus tu mets un grand nombre, plus l'image sera longue à générer",
        value=1,
    )
    nb_caps_cols = st.number_input(
        label="Nombre de capsules que tu veux mettres pour faire la largeur de l'image",
        value=20,
    )
    image_width = st.number_input(
        label="Nombre de pixels de largeur de l'image générée",
        value=2000,
    )
    mode = st.selectbox(
        label="Mode de remplissage",
        options=["Depuis le centre", "Double parcours de grille"],
    )
    if mode == "Depuis le centre":
        noise = st.number_input(
            label="Noise",
            value=100,
        )
    st.success("Tout va être calculé à partir des algorithmes de Samuel")

st.title("Bottle cap art")

uploaded_file = st.file_uploader(
    "Choisi une image, clique sur le bouton et regarde le résultat",
    accept_multiple_files=False
)

if uploaded_file is None:
    st.stop()

input_img = Image.open(uploaded_file)
with st.expander("Ton image", expanded=True):
    st.image(input_img, use_column_width=True)

button = st.button("Générer une version en capsules")
if button:
    output_img, df_caps_html = generate_image(
        input_img=input_img,
        nb_rotations=nb_rotations,
        nb_caps_cols=nb_caps_cols,
        image_width=image_width,
        count_limit=count_limit,
        mode=mode,
        noise=noise if mode == "Depuis le centre" else None,
        from_streamlit=True,
    )
    with st.expander("Ton image en capsules", expanded=True):
        st.image(output_img, use_column_width=True)
    with st.expander("Liste des capsules", expanded=True):
        st.markdown(df_caps_html, unsafe_allow_html=True)
