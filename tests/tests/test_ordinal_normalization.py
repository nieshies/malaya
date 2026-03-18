import unittest  
import os  
  
# Set up environment before importing malaya  
os.environ['CUDA_VISIBLE_DEVICES'] = ''  
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'  
  
import malaya  
  
  
class TestDateNormalization(unittest.TestCase):  
    """Test date normalization with ordinal days in English"""  
      
    @classmethod  
    def setUpClass(cls):  
        """Set up the normalizer once for all tests"""  
        cls.normalizer = malaya.normalize.normalizer()  
      
    def test_ordinal_date_range_english(self):  
        """Test ordinal days in date range with English output"""  
        input_text = 'Your student pass will be valid from 1 January 2024 to 31 December 2025.'  
        expected = 'Your student pass will be valid from first January two thousand and twenty four to thirty-first December two thousand and twenty five.'  
          
        result = self.normalizer.normalize(  
            input_text,  
            normalize_in_english=True,  
            translator=None,  
            language_detection_word=None,  
            normalize_text=False,  
            normalize_word_rules=False,  
            normalize_entity=False  
        )  
          
        # Verify the result structure  
        self.assertIsInstance(result, dict)  
        self.assertIn('normalize', result)  
        self.assertIn('date', result)  
        self.assertIn('money', result)  
          
        # Verify the normalized output matches expected  
        actual = result['normalize']  
        self.assertEqual(actual, expected)  
  
  
if __name__ == '__main__':  
    unittest.main()