import unittest  
import os  
  
# Set up environment before importing malaya  
os.environ['CUDA_VISIBLE_DEVICES'] = ''  
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'  
  
import malaya  
  
  
class TestPassportNormalization(unittest.TestCase):  
    """Test passport number normalization with malaya.normalizer.rules.load()"""  
      
    @classmethod  
    def setUpClass(cls):  
        """Set up the normalizer once for all tests using the enhanced load method"""  
        # Load language model for spelling correction  
        lm = malaya.language_model.kenlm(model = 'bahasa-wiki-news')  
          
        # Load speller and stemmer as shown in documentation  
        corrector = malaya.spelling_correction.probability.load(language_model = lm)  
        stemmer = malaya.stem.huggingface()  
          
        # Use the enhanced load method with corrector and stemmer  
        cls.normalizer = malaya.normalizer.rules.load(corrector, stemmer)  
      
    def test_passport_english_format(self):  
        """Test passport normalization: 'yes, may I repeat, your passport is EL9568719'"""  
        input_text = 'yes, may I repeat, your passport is EL9568719'  
          
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
            normalize_in_english=True,  
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
          
        # Verify passport was processed (should contain spoken digits)  
        self.assertIn('passport', normalized_text.lower())  
      
    def test_passport_malay_format(self):  
        """Test passport normalization: 'ini adalah passport saya A2096457'"""  
        input_text = 'ini adalah passport saya A2096457'  
          
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
            normalize_in_english=True,  
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
          
        # Verify passport was processed (should contain spoken digits)  
        self.assertIn('passport', normalized_text.lower())  
  
  
if __name__ == '__main__':  
    unittest.main()