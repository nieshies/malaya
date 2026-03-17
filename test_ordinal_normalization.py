import os  
  
os.environ['CUDA_VISIBLE_DEVICES'] = ''  
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'  
  
import malaya  
  
def test_date_normalization():  
      
    normalizer = malaya.normalize.normalizer()  
      
    test_cases = [  
        {  
            'input': 'Your student pass will be valid from 1 January 2024 to 31 December 2025.',  
            'expected': 'Your student pass will be valid from first January two thousand and twenty four to thirty-first December two thousand and twenty-five.',  
            'description': 'Ordinal days in date range (English)'  
        }  
    ]  
      
    for i, test in enumerate(test_cases, 1):  
        print(f"Input:    {test['input']}")  
        print(f"Expected: {test['expected']}")  
          
        # Use all parameters to ensure English output  
        result = normalizer.normalize(  
    test['input'],  
    normalize_in_english=True,  
    translator=None,  
    language_detection_word=None,  
    normalize_text=False,  # Disable text normalization  
    normalize_word_rules=False,  # Disable Malay word rules  
    normalize_entity=False  
)
        got = result['normalize']  
        print(f"Got:      {got}")  
          
if __name__ == "__main__":  
    test_date_normalization()