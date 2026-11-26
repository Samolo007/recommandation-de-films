import os

import requests
import streamlit as st

api_url = os.environ.get("API_URL", "http://127.0.0.1:8000")
st.title("Recommandation de films")


@st.cache_data
def load_movies():
    response = requests.get(f"{api_url}/movies", timeout=5)
    response.raise_for_status()
    return response.json()


try:
    movies = load_movies()
except requests.RequestException:
    st.error("Impossible de joindre l'API. Vérifiez qu'elle est lancée.")
    st.stop()

labels = {f"{movie['title']} ({movie['item_id']})": movie["item_id"] for movie in movies}

with st.form("choix"):
    choice = st.selectbox("Choisissez un film", list(labels))
    manual_id = st.number_input(
        "Ou saisissez un identifiant (laisser 0 pour utiliser la liste)",
        min_value=0,
        value=0,
        step=1,
    )
    submitted = st.form_submit_button("Recommander")

if submitted:
    item_id = int(manual_id) if manual_id > 0 else labels[choice]
    try:
        response = requests.post(f"{api_url}/predict", json={"item_id": item_id}, timeout=5)
    except requests.RequestException:
        st.error("Impossible de joindre l'API. Vérifiez qu'elle est lancée.")
        st.stop()

    body = response.json()
    if response.status_code == 200:
        st.subheader(f"Films proches de : {body['title']}")
        if body["recommendations"]:
            for rec in body["recommendations"]:
                st.write(f"- {rec['title']} (similarité {rec['score']:.2f})")
        else:
            st.info(body["message"])
    elif response.status_code == 404:
        st.error(body["detail"])
    else:
        st.error(f"Entrée invalide : {body['detail'][0]['msg']}")