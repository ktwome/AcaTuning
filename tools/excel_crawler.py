from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv
from os import getenv
import time

def click(class_name):
    # 통합정보시스템 접속
    # 명시적 대기 추가
    element = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.CLASS_NAME, class_name)))
    element.click() 

# 환경 변수 로드
load_dotenv()

dju_portal_url = 'https://portal.dju.ac.kr/'

# 크롬 드라이버 실행 및 url 실행
driver = webdriver.Chrome()
driver.get(dju_portal_url)


if input('o') == 'o':
    # 로그인 프로세스
    # ID 입력
    login_id_box = driver.find_element(By.CLASS_NAME, 'id_input1')
    login_id_box.send_keys(getenv('PORTAL_ID'))

    # 패스워드 입력
    login_id_box = driver.find_element(By.CLASS_NAME, 'id_input2')
    login_id_box.send_keys(getenv('PORTAL_PASSWORD'))

    driver.find_element(By.CLASS_NAME, 'loginbtn').click()
    time.sleep(1)

    # 통합정보시스템 접속
    driver.get('https://itics.dju.ac.kr/sso/sso.jsp')
    time.sleep(2)

    # 강좌개설조회 페이지 접속
    driver.find_element(By.ID, 'mainframe_childframe_form_leftContentDiv_widType_BTN_SEARCH_MENU_DIV_menuDiv_DG_LEFT_MENU_body_gridrow_2').click()
    time.sleep(2)

    # 대학 드롭다운 클릭
    driver.find_element(By.ID, 'mainframe_childframe_form_mainContentDiv_workDiv_WINB012902_SEARCHDIV01_S_HAKGWA_DaehakCombo_dropbuttonImageElement').click()
    time.sleep(1)

    # 전체 클릭릭
    driver.find_element(By.ID, 'mainframe_childframe_form_mainContentDiv_workDiv_WINB012902_SEARCHDIV01_S_HAKGWA_DaehakCombo_combolist_itemTextBoxElement').click()
    time.sleep(1)

    # 강의계획서 드롭다운 클릭
    driver.find_element(By.ID, 'mainframe_childframe_form_mainContentDiv_workDiv_WINB012902_SEARCHDIV01_S_PLANSEO_GB_Combo00_dropbuttonImageElement').click()
    time.sleep(1)

    # 전체 클릭릭
    # 두 번째 항목(작성) 선택
    second_item = driver.find_element(By.XPATH, "(//div[@id='mainframe_childframe_form_mainContentDiv_workDiv_WINB012902_SEARCHDIV01_S_PLANSEO_GB_Combo00_combolist_item'])[2]")
    second_item.click()
    time.sleep(0.5)

    # 조회 클릭
    driver.find_element(By.ID, 'mainframe_childframe_form_mainContentDiv_workDiv_WINB012902_buttonDiv_searchButtonTextBoxElement').click()
    time.sleep(2)
#936
    # 일단 해보자
    if input('o') == 'o':
        lecture_volume = 1023
        iter_number= (lecture_volume//4) + 1
        for _ in range(iter_number):
            for i in range(0, 40):
                driver.find_element(By.ID, f'mainframe_childframe_form_mainContentDiv_workDiv_WINB012902_DG_GRID01_body_gridrow_{i}_cell_{i}_10_controlbuttonTextBoxElement').click()
                time.sleep(3)

                driver.find_element(By.ID, 'mainframe_childframe_hl_4040306_p01_form_PopupForm_COMMONDIV01_UbiToolbar_toolbarDiv_EXPORT_DIV_BTN_SAVE').click()
                time.sleep(0.3)

                driver.find_element(By.ID, 'mainframe_childframe_hl_4040306_p01_form_PopupForm_COMMONDIV01_UbiToolbar_SAVE_LIST_saveExcelTextBoxElement').click()
                time.sleep(0.3)

                driver.find_element(By.ID, 'mainframe_childframe_hl_4040306_p01_form_titlebar_closebutton').click()
                time.sleep(0.5)
            driver.find_element(By.ID, 'mainframe_childframe_form_topDiv_titleDiv_BTN_TIMEOUT_RESETTextBoxElement').click()
            time.sleep(2)
            driver.switch_to.alert.accept()
            time.sleep(3)

    time.sleep(300)