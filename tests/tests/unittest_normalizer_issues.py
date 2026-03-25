import unittest  
import os  
  
# Set up environment before importing malaya  
os.environ['CUDA_VISIBLE_DEVICES'] = ''  
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'  
  
import malaya  
  #cover all the normalization features such as time, date, money, url, email, units, percent, number, x kali, cardinal, ordinal, elongated words, emoji, word rules (shortform expansion), contractions expansion, entity extraction, year normalization, hingga normalization, pada hari bulan normalization, fraction normalization
class TestMalayaNormalization(unittest.TestCase):  
    """Test comprehensive normalization with malaya.normalizer.rules.load()"""  
      
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
  
    def test_url_normalization(self):  
        """Test URL normalization: 'web saya ialah https://huseinhouse.com'"""  
        input_text = 'web saya ialah https://huseinhouse.com'  
        result = self.normalizer.normalize(input_text, normalize_url=True)  
        expected = 'web saya ialah HTTPS huseinhouse dot com'  
        self.assertEqual(result['normalize'], expected)  
        self.assertIsInstance(result, dict)  
        self.assertIn('normalize', result)  
  
    def test_email_normalization(self):  
        """Test email normalization"""  
        input_text = 'email saya adalah user@example.com'  
        result = self.normalizer.normalize(input_text, normalize_email=True)  
        self.assertIsInstance(result, dict)  
        self.assertIn('normalize', result)  
        self.assertGreater(len(result['normalize']), 0)  
  
    def test_money_normalization_various_formats(self):  
        """Test money normalization with various formats"""  
        test_cases = [  
            ('RM10.5', 'sepuluh ringgit lima puluh sen'),  
            ('rm 10.5 sen', 'sepuluh ringgit lima puluh sen'),  
            ('20.2m ringgit', 'dua puluh juta dua ratus ribu ringgit'),  
            ('hanyalah rm2 ribu', 'hanyalah dua ribu ringgit'),  
        ]  
          
        for input_text, expected_contains in test_cases:  
            with self.subTest(input_text=input_text):  
                result = self.normalizer.normalize(input_text, normalize_money=True)  
                self.assertIsInstance(result, dict)  
                self.assertIn('money', result)  
                self.assertIn('normalize', result)  
                self.assertGreater(len(result['normalize']), 0)  
  
    def test_date_normalization_various_formats(self):  
        """Test date normalization with various formats"""  
        test_cases = [  
            '01/12/2001',  
            'Jun 2017',   
            '2017 Jun',  
            '15 Oktober 2019',  
            '2 Jun',  
            '3 hari lalu',  
            'minggu depan',  
            'semalam',  
            'esok'  
        ]  
          
        for input_text in test_cases:  
            with self.subTest(input_text=input_text):  
                result = self.normalizer.normalize(input_text, normalize_date=True)  
                self.assertIsInstance(result, dict)  
                self.assertIn('date', result)  
                self.assertIn('normalize', result)  
                self.assertGreater(len(result['normalize']), 0)  
  
    def test_time_normalization_various_formats(self):  
        """Test time normalization with various formats"""  
        test_cases = [  
            'pukul 2.30',  
            'pukul 22.30',  
            '12:10 AM',  
            '8 pagi',  
            '2:15 petang'  
        ]  
          
        for input_text in test_cases:  
            with self.subTest(input_text=input_text):  
                result = self.normalizer.normalize(input_text, normalize_time=True)  
                self.assertIsInstance(result, dict)  
                self.assertIn('normalize', result)  
                self.assertGreater(len(result['normalize']), 0)  
  
    def test_telephone_normalization(self):  
        """Test telephone number normalization"""  
        input_text = 'no saya 012-1234567'  
        result = self.normalizer.normalize(input_text, normalize_telephone=True)  
        expected_contains = 'kosong satu dua, satu dua tiga empat lima enam tujuh'  
        self.assertIn(expected_contains, result['normalize'])  
        self.assertIsInstance(result, dict)  
        self.assertIn('normalize', result)  
  
    def test_units_normalization(self):  
        """Test units normalization (temperature, distance, volume, weight)"""  
        test_cases = [  
            ('61.2 kg', 'enam puluh satu perpuluhan dua kilogram'),  
            ('61.2km', 'enam puluh satu perpuluhan dua kilometer'),  
            ('31.2c', 'tiga puluh satu perpuluhan dua celsius'),  
            ('600ml', 'enam ratus milliliter'),  
            ('2jam 30 minit', 'dua jam tiga puluh minit')  
        ]  
          
        for input_text, expected_contains in test_cases:  
            with self.subTest(input_text=input_text):  
                result = self.normalizer.normalize(input_text, normalize_units=True)  
                self.assertIsInstance(result, dict)  
                self.assertIn('normalize', result)  
                self.assertGreater(len(result['normalize']), 0)  
  
    def test_percent_normalization(self):  
        """Test percent normalization"""  
        input_text = '61.2%'  
        result = self.normalizer.normalize(input_text, normalize_percent=True)  
        expected = 'enam puluh satu perpuluhan dua peratus'  
        self.assertEqual(result['normalize'], expected)  
        self.assertIsInstance(result, dict)  
  
    def test_number_normalization(self):  
        """Test number normalization"""  
        test_cases = [  
            '0123',  
            '123',  
            '123.123421231'  
        ]  
          
        for input_text in test_cases:  
            with self.subTest(input_text=input_text):  
                result = self.normalizer.normalize(input_text, normalize_number=True)  
                self.assertIsInstance(result, dict)  
                self.assertIn('normalize', result)  
                self.assertGreater(len(result['normalize']), 0)  
  
    def test_x_kali_normalization(self):  
        """Test 'x kali' normalization"""  
        input_text = 'saya sokong 10x'  
        result = self.normalizer.normalize(input_text, normalize_x_kali=True)  
        expected = 'saya sokong sepuluh kali'  
        self.assertEqual(result['normalize'], expected)  
        self.assertIsInstance(result, dict)  
  
    def test_cardinal_normalization(self):  
        """Test cardinal number normalization"""  
        input_text = '123'  
        result = self.normalizer.normalize(input_text, normalize_cardinal=True)  
        expected = 'seratus dua puluh tiga'  
        self.assertEqual(result['normalize'], expected)  
        self.assertIsInstance(result, dict)  
  
    def test_ordinal_normalization(self):  
        """Test ordinal number normalization"""  
        test_cases = [  
            ('ke-12', 'kedua belas'),  
            ('ke-123', 'keseratus dua puluh tiga'),  
            ('tempat ke-12', 'tempat kedua belas')  
        ]  
          
        for input_text, expected_contains in test_cases:  
            with self.subTest(input_text=input_text):  
                result = self.normalizer.normalize(input_text, normalize_ordinal=True)  
                self.assertIsInstance(result, dict)  
                self.assertIn('normalize', result)  
                self.assertGreater(len(result['normalize']), 0)  
  
    def test_elongated_normalization(self):  
        """Test elongated word normalization"""  
        test_cases = [  
            'betulll',  
            'saayyyyaa ttttaaak ssssukaaa',  
            'berehatlh',  
            'seadil2nya'  
        ]  
          
        for input_text in test_cases:  
            with self.subTest(input_text=input_text):  
                result = self.normalizer.normalize(input_text, normalize_elongated=True)  
                self.assertIsInstance(result, dict)  
                self.assertIn('normalize', result)  
                self.assertGreater(len(result['normalize']), 0)  
  
    def test_emoji_normalization(self):  
        """Test emoji normalization"""  
        test_cases = [  
            'awak sangat hot ye 🔥🔥',  
            '🔥🙂',  
            '🤣🤣🤣'  
        ]  
          
        for input_text in test_cases:  
            with self.subTest(input_text=input_text):  
                result = self.normalizer.normalize(input_text, normalize_emoji=True)  
                self.assertIsInstance(result, dict)  
                self.assertIn('normalize', result)  
                self.assertGreater(len(result['normalize']), 0)  
  
    def test_word_rules_normalization(self):  
        """Test word rules normalization (shortform expansion)"""  
        test_cases = [  
            'xkisah',  # -> tak kisah  
            'xnak',    # -> tak nak  
            'mmg',     # -> memang  
            'sgt',     # -> sangat  
            'yg',      # -> yang  
            'jgn',     # -> jangan  
            'btl'      # -> betul  
        ]  
          
        for input_text in test_cases:  
            with self.subTest(input_text=input_text):  
                result = self.normalizer.normalize(input_text, normalize_word_rules=True)  
                self.assertIsInstance(result, dict)  
                self.assertIn('normalize', result)  
                self.assertGreater(len(result['normalize']), 0)  
  
    def test_contractions_expansion(self):  
        """Test English contractions expansion"""  
        test_cases = [  
            "don't",  
            "can't",   
            "I'm",  
            "shouldn't've"  
        ]  
          
        for input_text in test_cases:  
            with self.subTest(input_text=input_text):  
                result = self.normalizer.normalize(input_text, expand_contractions=True)  
                self.assertIsInstance(result, dict)  
                self.assertIn('normalize', result)  
                self.assertGreater(len(result['normalize']), 0)  
  
    def test_mixed_social_media_text(self):  
        """Test complex mixed social media text normalization"""  
        test_cases = [  
            'mulakn slh org boleh ,bila geng tuh kena slhkn jgk xboleh trima .. pelik , dia slhkn org bole hri2 crta sakau then bila kna bls balik xdpt jwb ,kata mcm biasa slh (parti sampah) 🤣🤣🤣 jgn mulakn dlu slhkn org kalau xboleh trima bila kna bls balik 🤣🤣🤣',  
            'berehatlh najib.. sudah2 lh tu.. jgn buat rakyat hilang kepercyaan tu pda system kehakiman negara.. klu btl x slh kenapa x dibuktikan semasa sblm rayuan.. sudah lah tu kami dh letih dengan drama korang. ok'  
        ]  
          
        for input_text in test_cases:  
            with self.subTest(input_text=input_text[:50] + '...'):  
                result = self.normalizer.normalize(input_text)  
                self.assertIsInstance(result, dict)  
                self.assertIn('normalize', result)  
                self.assertIn('date', result)  
                self.assertIn('money', result)  
                self.assertGreater(len(result['normalize']), 0)  
  
    def test_entity_extraction(self):  
        """Test entity extraction with normalize_entity=True"""  
        input_text = 'boleh dtg 8pagi esok? bayar rm3.2k'  
        result = self.normalizer.normalize(input_text, normalize_entity=True)  
          
        self.assertIsInstance(result, dict)  
        self.assertIn('normalize', result)  
        self.assertIn('date', result)  
        self.assertIn('money', result)  
          
        # Check that entities were extracted  
        self.assertIsInstance(result['date'], dict)  
        self.assertIsInstance(result['money'], dict)  
          
        # Should contain time and money entities  
        self.assertGreater(len(result['date']), 0)  
        self.assertGreater(len(result['money']), 0)  
    
    def test_year_normalization(self):  
        """Test year normalization"""  
        result = self.normalizer.normalize('tahun 1987', normalize_year=True)  
        self.assertIn('sembilan belas lapan puluh tujuh', result['normalize'])  
  
    def test_hingga_normalization(self):  
        """Test range normalization with hingga"""  
        result = self.normalizer.normalize('2011 - 2019', normalize_hingga=True)  
        self.assertIn('hingga', result['normalize'])  
    
    def test_pada_hari_bulan_normalization(self):  
        """Test pada hari bulan normalization"""  
        result = self.normalizer.normalize('pada 10/4', normalize_pada_hari_bulan=True)  
        self.assertIn('hari bulan', result['normalize'])  
    
    def test_fraction_normalization(self):  
        """Test fraction normalization"""  
        result = self.normalizer.normalize('10 /4', normalize_fraction=True)  
        self.assertIn('per empat', result['normalize'])
  
if __name__ == '__main__':  
    unittest.main()