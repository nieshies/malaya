import unittest  
import os  
  
# Set up environment before importing malaya  
os.environ['CUDA_VISIBLE_DEVICES'] = ''  
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'  
  
import malaya  
  
  
class TestNormalization(unittest.TestCase):  
    """Test phone, number, and time normalization in malaya.normalize.normalizer()"""  
      
    @classmethod  
    def setUpClass(cls):  
        """Set up the normalizer once for all tests"""  
        cls.normalizer = malaya.normalize.normalizer()  
      
    def test_phone_number_normalization(self):  
        """Test phone number normalization: '03-2782 5300' to words"""  
        input_text = 'The EMGS hotline, 03-2782 5300 is available for enquiries from 9:00 a.m. to 5:00 p.m. on weekdays.'  
        expected = 'The EMGS hotline, zero three - two seven eight two five three zero zero is available for enquiries from at nine am to at five pm on weekdays.'  
          
        result = self.normalizer.normalize(  
            input_text,  
            normalize_in_english=True,  
            translator=None,  
            language_detection_word=None,  
            normalize_text=False,  
            normalize_cardinal=False,  
            normalize_ordinal=False,  
            normalize_ic=True,  
            ic_dash_sempang=False,  
            normalize_hingga=False,  
            normalize_number=True,  
            normalize_telephone=True,  
        )  
          
        # Verify the result structure  
        self.assertIsInstance(result, dict)  
        self.assertIn('normalize', result)  
        self.assertIn('date', result)  
        self.assertIn('money', result)  
          
        # Verify the normalized output matches expected  
        actual = result['normalize']  
        self.assertEqual(actual, expected)  
      
    def test_number_normalization(self):  
        """Test number normalization: '59100' to words"""  
        input_text = 'The EMGS panel clinic at 59100 Kuala Lumpur is open from 9:00 a.m. to 5:00 p.m. on weekdays.'  
        expected = 'The EMGS panel clinic at five nine one zero zero Kuala Lumpur is open from at nine am to at five pm on weekdays.'  
          
        result = self.normalizer.normalize(  
            input_text,  
            normalize_in_english=True,  
            translator=None,  
            language_detection_word=None,  
            normalize_text=False,  
            normalize_cardinal=False,  
            normalize_ordinal=False,  
            normalize_ic=True,  
            ic_dash_sempang=False,  
            normalize_hingga=False,  
            normalize_number=True,  
            normalize_telephone=True,  
        )  
          
        # Verify the result structure  
        self.assertIsInstance(result, dict)  
        self.assertIn('normalize', result)  
        self.assertIn('date', result)  
        self.assertIn('money', result)  
          
        # Verify the normalized output matches expected  
        actual = result['normalize']  
        self.assertEqual(actual, expected)  
      
    def test_time_normalization(self):  
        """Test time normalization: '2.15 pm' to '14.15'"""  
        input_text = 'User will see you around 2.15 pm'  
        expected = 'User will see you around at 14.15'  
          
        result = self.normalizer.normalize(  
            input_text,  
            normalize_in_english=True,  
            translator=None,  
            language_detection_word=None,  
            normalize_text=False,  
            normalize_cardinal=False,  
            normalize_ordinal=False,  
            normalize_ic=True,  
            ic_dash_sempang=False,  
            normalize_hingga=False,  
            normalize_number=True,  
            normalize_telephone=True,  
            normalize_time=False,  
            normalize_entity=True,  
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