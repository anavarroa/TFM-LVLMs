import unittest
import time

class TestClass(unittest.TestCase):
    
    def setUp(self):
        self.class_to_test = Class()

    def tearDown(self):
        time.sleep(2)
        del self.class_to_test

    def test_1_class(self):
        
        input = "Whatever"
        result1 = self.class_to_test.method_1(input)
        self.assertEqual(result1, 3)

if __name__ == '__main__':
    unittest.main()