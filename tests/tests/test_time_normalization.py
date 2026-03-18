import os  
  
os.environ['CUDA_VISIBLE_DEVICES'] = ''  
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'  
  
import malaya  
  
# Load the normalizer  
normalizer = malaya.normalize.normalizer()  
  
# Test Case 1: Phone Number   
print("test case 1 :")  
input_1 = 'The EMGS hotline, 03-2782 5300 is available for enquiries from 9:00 a.m. to 5:00 p.m. on weekdays.'  
expected_1 = 'The E M G S hotline, zero three two seven eight two five three zero zero is available for enquiries from nine a m to five pm on weekdays.'  
  
result_1 = normalizer.normalize(  
    input_1,  # Fixed: was test['input']  
    normalize_in_english=True,  
    translator=None,  
    language_detection_word=None,  
    normalize_text=False,
    normalize_cardinal=False,
    normalize_ordinal=False,
    normalize_ic=True,
    ic_dash_sempang= False,
    normalize_hingga= False,
    normalize_number=True, 
    normalize_telephone=True,
)  
  
print("input :", input_1)  
print("expected :", expected_1)  
print("got :", result_1['normalize'])  
print()  
  
# Test Case 2: Number Normalization  
print("test case 2 :")  
input_2 = 'The EMGS panel clinic at 59100 Kuala Lumpur is open from 9:00 a.m. to 5:00 p.m. on weekdays.'  
expected_2 = 'The E M G S panel clinic at five nine one zero zero, Kuala Lumpur, is open from nine a m to five pm on weekdays.'  
  
result_2 = normalizer.normalize(  
    input_2,  # Fixed: was test['input']  
    normalize_in_english=True,  
    translator=None,  
    language_detection_word=None,  
    normalize_text=False,
    normalize_cardinal=False,
    normalize_ordinal=False,
    normalize_ic=True,
    ic_dash_sempang= False,
    normalize_hingga= False,
    normalize_number=True, 
    normalize_telephone=True, 
)  
  
print("input :", input_2)  
print("expected :", expected_2)  
print("got :", result_2['normalize'])  
print()  
  
# Test Case 3: Time Normalization  
print("test case 3 :")  
input_3 = 'User will see you around 2.15 pm'  
expected_3 = 'User will see you around at 14.15'  
  
result_3 = normalizer.normalize(  
    input_3,  # Fixed: was test['input']  
    normalize_in_english=True,  
    translator=None,  
    language_detection_word=None,  
    normalize_text=False,
    normalize_cardinal=False,
    normalize_ordinal=False,
    normalize_ic=True,
    ic_dash_sempang= False,
    normalize_hingga= False,
    normalize_number=True, 
    normalize_telephone=True,
    normalize_time=False,
    normalize_entity=True,
)  
  
print("input :", input_3)  
print("expected :", expected_3)  
print("got :", result_3['normalize'])
