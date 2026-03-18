import os  
  
os.environ['CUDA_VISIBLE_DEVICES'] = ''  
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'  
  
import malaya  
  
def test_date_normalization():  
    # Use the correct normalizer loader  
    normalizer = malaya.normalize.normalizer()  
      
    test_cases = [  
        {  
            'input': "Malaysia's National Day is on 31 August",  
        },  
        {  
            'input': "Malaysia's National Day is on December 31",  
        }  
    ]  
      
    for i, test in enumerate(test_cases, 1):  
        print(f"Test Case {i}")  
        print(f"Input:    {test['input']}")  
          
        result = normalizer.normalize(  
            test['input'],  
            normalize_in_english=True,  
            translator=None,  
            language_detection_word=None,  
            normalize_text=False,  
            normalize_word_rules=False,  
            normalize_entity=True,  
            normalize_date=True  
        )  
        got = result['normalize']  
        print(f"Got:      {got}")  
        print()  
          
if __name__ == "__main__":  
    test_date_normalization()