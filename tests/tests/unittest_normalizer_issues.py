import unittest  
import os  
  
# Set up environment before importing malaya  
os.environ['CUDA_VISIBLE_DEVICES'] = ''  
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'  
  
import malaya  
  
  
class TestMalayaNormalization(unittest.TestCase):  
    """Test comprehensive normalization with malaya.normalize.normalizer()"""  
      
    @classmethod  
    def setUpClass(cls):  
        """Set up the normalizer once for all tests"""  
        cls.normalizer = malaya.normalize.normalizer()  
      
    def test_time_normalization_pukul_8(self):  
        """Test time normalization: 'pukul 8'"""  
        input_text = 'pukul 8'  
          
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
          
        # Verify the normalized text is not empty  
        normalized_text = result['normalize']  
        self.assertIsInstance(normalized_text, str)  
        self.assertGreater(len(normalized_text), 0)  
      
    def test_unit_and_date_normalization(self):  
        """Test unit number and date normalization: 'Sila hantar ke unit 803, tingkat 8, sebelum pukul 5, pada 10hb'"""  
        input_text = 'Sila hantar ke unit 803, tingkat 8, sebelum pukul 5, pada 10hb'  
          
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
      
    def test_booking_date_time_normalization(self):  
        """Test booking with date and time: 'Tempahan untuk 4 orang, pada 12 07, meja nombor 18, pukul 8 malam.'"""  
        input_text = 'Tempahan untuk 4 orang, pada 12 07, meja nombor 18, pukul 8 malam.'  
          
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
      
    def test_money_date_time_account_normalization(self):  
        """Test money, date, time, and account number: 'Saya dah bayar bil elektrik sebanyak RM 87.45 pada 15 Jun, pukul 2:15 petang, untuk akaun nombor 120387.'"""  
        input_text = 'Saya dah bayar bil elektrik sebanyak RM 87.45 pada 15 Jun, pukul 2:15 petang, untuk akaun nombor 120387.'  
          
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
          
        # Verify money was extracted (RM 87.45 should be in money dict)  
        self.assertIsInstance(result['money'], dict)  
          
        # Verify the normalized text contains expected elements  
        normalized_text = result['normalize']  
        self.assertIsInstance(normalized_text, str)  
        self.assertGreater(len(normalized_text), 0)  
      
    def test_ic_number_normalization(self):  
        """Test IC number normalization: 'Nombor IC saya adalah 000-11937740-18818tt3383333000000'"""  
        input_text = 'Nombor IC saya adalah 000-11937740-18818tt3383333000000'  
          
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
  
  
if __name__ == '__main__':  
    unittest.main()