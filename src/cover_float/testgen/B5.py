#Lamarr
#B5
#This model creates a test-case for each of the following constraints on the intermediate results:
#-------- Tests Required --------
#1) A random positive SubNorm
#2) A random negative SubNorm
#3) All the numbers in the range [+MinSubNorm - 3 ulp, +MinSubNorm + 3 ulp]
#4) All the numbers in the range [-MinSubNorm - 3, -MinSubNorm + 3 ulp]
#5) All the numbers in the range [MinNorm - 3 ulp, MinNorm + 3 ulp]
#6) All the numbers in the range [-MinNorm - 3 ulp, -MinNorm + 3 ulp]
#7) A random number in the range (0, MinSubNorm)
#8) A random number in the range (-MinSubNorm, -0)
#9) One number for every exponent in the range [MinNorm.exp, MinNorm.exp +5]
#-------- Definitions --------
#  Branch: The implementation of certain restrictions towards Base and Dynamic based on the test, operation, and their signs
#  Branch naming sequence: operation_sign(Base)_sign(Dynamic) ex - a_pp = addition, where base and sign are positive
#  maxSN: the largest subnorm value (based on precision)
#  base_mant: The mantissa of the Bantissa
#  dynamic_mant: The mantissa of the Dynamic
#-------- Test 1 --------
#Operation: Addition
#    After establishing the type of operation, the restriction of Base and Dynamic are based on their respective signs
#    The signs of Base and Dynamic will be randomly generated
#    Based on the random value determined, there will be 4 branches
#    These branches include: a_pp, a_pn, a_np, a_nn
#    a_pp:
#        base range = [1, maxSN - 1]
#        dynamic range = [1, maxSN - base_mant]
#        base_exp = 0
#        dynamic_exp = 0
#    a_pn:
#        base range = []