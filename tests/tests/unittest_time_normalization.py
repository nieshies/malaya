import unittest  
import os  
  
# Set up environment before importing malaya  
os.environ['CUDA_VISIBLE_DEVICES'] = ''  
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'  
  
import malaya  
  
  
class TestTimeNormalization(unittest.TestCase):  
    """Test time normalization with malaya.normalize.normalizer()"""  
      
    @classmethod  
    def setUpClass(cls):  
        """Set up the normalizer once for all tests"""  
        cls.normalizer = malaya.normalize.normalizer()  
      
    def test_service_hours_normalization(self):  
        """Test service hours time normalization: 'Waktu perkhidmatan kami adalah dari pukul 9 pagi hingga 6 petang'"""  
        input_text = 'Waktu perkhidmatan kami adalah dari pukul 9 pagi hingga 6 petang'  
          
        result = self.normalizer.normalize(  
            input_text,  
            normalize_hingga=False,  
            normalize_text=True,  
            normalize_word_rules=False,  
            normalize_time=True,  
            normalize_cardinal=False,  
            normalize_ordinal=False,  
            normalize_url=True,  
            normalize_email=True,  
            normalize_in_english=False,  
        )  
          
        # Verify the result structure  
        self.assertIsInstance(result, dict)  
        self.assertIn('normalize', result)  
        self.assertIn('date', result)  
        self.assertIn('money', result)  
          
        # Verify the normalized text contains expected elements  
        normalized_text = result['normalize']  
        self.assertIsInstance(normalized_text, str)  
        self.assertGreater(len(normalized_text), 0)  
          
        # Verify time was processed (should contain 'pukul' and time expressions)  
        self.assertIn('pukul', normalized_text.lower())  
  
  
if __name__ == '__main__':  
    unittest.main()