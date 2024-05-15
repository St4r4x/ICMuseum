import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime


def download_image(url, output_dir):
    """
    Télécharge une image à partir de l'URL spécifiée et l'enregistre dans le répertoire spécifié.

    Args:
        url (str): L'URL de l'image à télécharger.
        output_dir (str): Le répertoire dans lequel enregistrer l'image téléchargée.

    Returns:
        None
    """
    response = requests.get(url)
    if response.status_code == 200:
        image_name = os.path.basename(url)
        output_path = os.path.join(output_dir, image_name)

        # Vérifie si le répertoire existe, sinon le crée
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Image {image_name} téléchargée avec succès.")
    else:
        print(f"Échec du téléchargement de l'image {url}.")

def extract_images(base_url, year, month, output_dir):
    """
    Extrait et télécharge toutes les images à partir de l'URL spécifiée.

    Args:
        base_url (str): L'URL de base à partir de laquelle extraire les images.
        year (str): L'année à inclure dans l'URL.
        month (str): Le mois à inclure dans l'URL.
        output_dir (str): Le répertoire dans lequel enregistrer les images téléchargées.

    Returns:
        None
    """
    url = f"{base_url}/{year}/{month}/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a')
        for link in links:
            link_url = urljoin(url, link.get('href'))
            if link_url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                download_image(link_url, output_dir)
    else:
        print(f"Échec de la requête GET à l'URL {url}.")

if __name__ == "__main__":
    base_url = 'https://www.musee-fesch.com/wp-content/uploads'
    output_dir = './database/'
    # Obtient l'année en cours
    current_year = datetime.now().year

    # Parcourt les années de 2020 à l'année en cours
    for year in range(2020, current_year + 1):
        # Parcourt les mois de 01 à 12
        for month in range(1, 13):
            # Formate le mois pour qu'il ait toujours deux chiffres
            month = str(month).zfill(2)
            # Appelle la fonction extract_images avec l'année et le mois courants
            extract_images(base_url, str(year), month, output_dir)