from joblib import Parallel, delayed
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException

driver = webdriver.Chrome()
driver.get("https://www.vgmusic.com/music/console/nintendo/nes/")

songs_table = driver.find_element_by_css_selector(" tbody")
song_names = []


def download_single_by_xpath(i):
    try:
        song = driver.find_element_by_xpath(f"/html/body/table[1]/tbody/tr[{i}]/td[1]/a")
        song_names.append(song.text)
    except NoSuchElementException:
        print("No such element")


result = Parallel(n_jobs=-1, verbose=2, backend="threading")(
    map(delayed(download_single_by_xpath), range(1, 1000)))

driver.close()

print(song_names)
