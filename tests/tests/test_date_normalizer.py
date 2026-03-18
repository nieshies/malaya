import unittest  
import os  
  
# Set up environment before importing malaya  
os.environ['CUDA_VISIBLE_DEVICES'] = ''  
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'  
  
import malaya  
  
  
class TestDateNormalization(unittest.TestCase):  
    """Test date normalization functionality in malaya.normalize.normalizer()"""  
      
    @classmethod  
    def setUpClass(cls):  
        """Set up the normalizer once for all tests"""  
        cls.normalizer = malaya.normalize.normalizer()  
      
    def test_date_normalization_august(self):  
        """Test normalization of '31 August' date format"""  
        test_input = "Malaysia's National Day is on 31 August"  
          
        result = self.normalizer.normalize(  
            test_input,  
            normalize_in_english=True,  
            translator=None,  
            language_detection_word=None,  
            normalize_text=False,  
            normalize_word_rules=False,  
            normalize_entity=True,  
            normalize_date=True  
        )  
          
        # Verify the result structure  
        self.assertIsInstance(result, dict)  
        self.assertIn('normalize', result)  
        self.assertIn('date', result)  
        self.assertIn('money', result)  
          
        # Verify date was extracted and parsed  
        self.assertIsNotNone(result['date'])  
        self.assertGreater(len(result['date']), 0)  
          
        # Verify the normalized text contains the date  
        normalized_text = result['normalize']  
        self.assertIsInstance(normalized_text, str)  
        self.assertGreater(len(normalized_text), 0)  
      
    def test_date_normalization_december(self):  
        """Test normalization of 'December 31' date format"""  
        test_input = "Malaysia's National Day is on December 31"  
          
        result = self.normalizer.normalize(  
            test_input,  
            normalize_in_english=True,  
            translator=None,  
            language_detection_word=None,  
            normalize_text=False,  
            normalize_word_rules=False,  
            normalize_entity=True,  
            normalize_date=True  
        )  
          
        # Verify the result structure  
        self.assertIsInstance(result, dict)  
        self.assertIn('normalize', result)  
        self.assertIn('date', result)  
        self.assertIn('money', result)  
          
        # Verify date was extracted and parsed  
        self.assertIsNotNone(result['date'])  
        self.assertGreater(len(result['date']), 0)  
          
        # Verify the normalized text contains the date  
        normalized_text = result['normalize']  
        self.assertIsInstance(normalized_text, str)  
        self.assertGreater(len(normalized_text), 0)  
  
  
if __name__ == '__main__':  
    unittest.main()