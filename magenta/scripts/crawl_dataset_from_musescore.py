from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json

chrome_driver_path = './chromedriver'
save_directory = 'donwloads'
download_extention = 'MusicXML' #Please refer to the text on download popup in musescore
username = 'ViMusic2019'
password = 'viralint@2019'

class musescore_comm:
    def __init__(self):
        self.main_page = 'http://www.musescore.com'

        self.username = username
        self.password = password

        self.download_extention = download_extention
        
        self.chrome_options = webdriver.ChromeOptions()
        
        self.chrome_options.add_experimental_option("prefs", {
            "download.default_directory": save_directory,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
        })        
        self.chrome_options.add_argument("--window-size=1920x1080")
        self.chrome_options.add_argument("--disable-notifications")
        self.chrome_options.add_argument('--headless')
        #self.chrome_options.add_argument('--no-sandbox') # required when running as root user. otherwise you would get no sandbox errors. 
                
        self.driver = None
        
        self.selected_instruments = []
        self.current_sorting_type = 'Relevance'
        
        self.scores_dict = {}
        
    def show_current_url(self):
        print('Current URL: ' + self.driver.current_url)
    
    def show_current_config(self):
        print('Current selected instruments: ' + '.'.join(self.selected_instruments) if '.'.join(self.selected_instruments) else 'None')
        print('Current sorting type: ' + self.current_sorting_type)

    def save_screenshot(self, name):
        self.driver.save_screenshot(name)
        
    def connect(self):
        print('Initializing chrome binary...')
        self.driver = webdriver.Chrome(executable_path=chrome_driver_path, options=self.chrome_options)
        
        print('Loading page: ' + self.main_page)
        self.driver.get(self.main_page)
        
        print('Login account ID: ' + self.username + ' - Password: ' + self.password)
        button = self.driver.find_element_by_class_name("login")
        button.click()
        username = self.driver.find_element_by_id("edit-name")
        username.send_keys(self.username)
        password = self.driver.find_element_by_id("edit-pass")
        password.send_keys(self.password)
        login = self.driver.find_element_by_id("edit-submit")
        login.click()

        print('Go to main score page')
        self.driver.get(self.main_page + '/sheetmusic')
        
    def login(self):
        pass
    
    def go_to_url(self, url):
        self.driver.get(url)
    
    def reconnect(self):
        self.driver.get('https://musescore.com/sheetmusic')

    def go_forward(self):
        self.driver.forward()
        
    def go_backward(self):
        self.driver.back()
        
    def get_selected_instruments(self):
        return self.selected_instruments
        
    def query_all_scores_url_in_current_filter(self, limit=None):
        '''Get all the scores in the current filter
            arg:
                limit: Limit the number of scores to get
        '''
        def _check_query_over_limit(current_dict, limit):
            state=False
            if limit is not None:
                if len(current_dict) >= limit:
                    print('Reached scores limit number')
                    state = True
                    
            return state
        
        if limit is not None:
            if limit % 20 != 0:
                print('Limit number must be divived by 20')
                return -1
        
        self.scores_dict = {}
        query_state = -1       
        
        #Query the url in first page
        query_state = self.query_all_scores_url_in_current_page()
            
        #Click next after done querying each page, until the last page
        while self._click_next_button() == 1:
            if _check_query_over_limit(self.scores_dict, limit):
                break
                
            if self.query_all_scores_url_in_current_page() == -1:
                break
                
        return 0
            
    def query_all_scores_url_in_current_page(self):
        query_state = -1
        
        try:
            #Find all scores showing in one page
            scores_title = self.driver.find_elements_by_class_name('score-info__title')
                    
            #Get the hyperlink from each score's title
            for title in scores_title:
                try:
                    tag = title.find_element_by_tag_name('a')
                    name = tag.text
                    url = tag.get_attribute('href')
                    self.scores_dict[name] = url
                    
                    print('Collected score url #' + str(len(self.scores_dict)) + ' with NAME : ' + name +  ' and URL: ' + url)
                    query_state = 1
                except:
                    print('Unable to find hyperlink in the score-s title')
        except:
            print('Unable to find any score')
            
        return query_state
    
    def get_name_by_index(self, index):
        return list(self.scores_dict)[index]
    
    def get_url_by_index(self, index):
        return list(self.scores_dict.values())[index]
    
    def save_scores_dict_to_json(self, path):
        with open(path, 'w') as fp:
            json.dump(self.scores_dict, fp)
    
    def load_scores_dict_from_json(self, path):
        with open(path, 'r') as fp:
            self.scores_dict = json.load(fp)
            
    def download_score_link(self, link):
        print('Go to url: ' + link)
        self.go_to_url(link)
        print('Unable to click download button') if self._click_download_button() == -1 else print('Clicked download button')
        #Wait until the popup window showed
        try:
            element = WebDriverWait(self.driver, 1).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "article._2O4bQ.oQvef._1byly"))
            )
        except:
            print('Extension popup not showing')
            return -1

        print('Unable to click extention button') if self._click_score_extension(self.download_extention) == -1 else print('Clicked extention button - Download proccessing')

    def _go_to_score_url(self, link):
        pass
    
    def _click_download_button(self):
        button_state = -1
        try:
            buttons_found = self.driver.find_elements(By.CSS_SELECTOR, 'span._3R0py')
            download_button = None
            for b in buttons_found:
                if b.text == 'Download':
                    download_button = b.find_element_by_xpath('..')
                    download_button.click()
                    button_state = 1
                    break
        except:
            pass

        return button_state

    def _click_score_extension(self, ext_type):
        button_state = -1
        
        try:
            buttons_found = self.driver.find_elements(By.CSS_SELECTOR, 'span._2Tbln')
            ext_button = None
            for b in buttons_found:
                if b.text == ext_type:
                    ext_button = b.find_element_by_xpath('../..')
                    ext_button.click()
                    button_state = 1
                    break
        except:
            pass

        return button_state
    
    def _click_next_button(self):
        ''' Try to find the next button
            return:  
                -1 = Next button doesn't exist
                 0 = Next button is disabled
                 1 = Next button found and clicked to next page
        '''
                
        next_button_state = -1
            
        try:
            #Find next button
            next_button = handler.driver.find_element(By.CSS_SELECTOR,"li.pager__item.next");
            next_button_tag = next_button.find_element_by_tag_name('a')
            next_button_url = next_button_tag.get_attribute('href')
            
            self.go_to_url(next_button_url)
            
            next_button_state = 1
        except:
            pass
        try:
            #Find next button
            next_button = handler.driver.find_element(By.CSS_SELECTOR,"li.pager__item.next.disabled");
            next_button_state = 0
        except:
            pass
        
        print('Next button doesnt exist in current page' if next_button_state == -1 else ('Next button is disabled' if next_button_state == 0 else 'Next button pressed'))
            
        return next_button_state
    
    def choose_instrument(self, instrument_name):        
        try:
            # Try to find the name in current selected instruments 
            
            # Found => Remove the name from selected instruments
            # Click the relevant button
            index = self.selected_instruments.index(instrument_name)

            print('Unselect instrument from filter:' + self._concat_string_filter(instrument_name))
            status = self._click_button_element('(-) ' + instrument_name)
            
            if status is True:
                self.selected_instruments.pop(index)
            
        except:
            # Not Found => Add the name to selected instruments
            # Click the relevant button
            
            print('Select instrument into filter:' + self._concat_string_filter(instrument_name))  
            status = self._click_button_element(instrument_name)
            
            if status is True:
                self.selected_instruments.append(instrument_name)

        print('Current selected instruments: ' + '.'.join(self.selected_instruments) if '.'.join(self.selected_instruments) else 'None')
  
    def _click_button_element(self, name):
        click_status = False
        
        try:
            self.driver.find_element(By.LINK_TEXT, name).click()
            click_status = True
        except:
            print('There is no button with the name: ' + name)
            
        return click_status
    
    def sorting_by(self, filter_type):
        status = self._click_button_element(filter_type) 

        if status is True:
            self.current_sorting_type = filter_type
        
        print('Current sorting type: ' + self.current_sorting_type)

    def go_to_next_page(self):
        pass
            
    def find_song_genre(self, name):
        pass
    
    def _concat_string_filter(self, string):
        string_changed = ' >> ' + string + ' << '
        return string_changed
         
if __name__== "__main__":
    handler = musescore_comm()
    handler.connect()
    handler.load_scores_dict_from_json('score.json')

    for index in range(30):
        handler.download_score_link(handler.get_url_by_index(index))