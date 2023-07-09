# BFV

## About BFV RNS Variants

Scheme 마다 구현한 BFV Variant가 다를 수 있음. (RNS를 구현하는 방식)

RNS = Optimization by using of CRT(Chinese Remainder Theorem) representation of the large integers

[The BGV and BFV Encryption Schemes](https://docs.google.com/document/d/1LS7cL7cG1RdfkGUYwObZks67CSVM6oOFU_SHTNYm2Nw/edit#heading=h.y5w4ljc08xkb)

> Two RNS variants of BFV are known in literature: **BEHZ16** (based on integer arithmetic) and **HPS19**(based on both integer and floating-point arithmetic). A comparison of the RNS variants is provided in **BPAVR19**.
>
> The encoding of vectors of integers into a BFV plaintext is described in Appendix A of [LN14]. This batching/packing encoding technique is discussed at a more advanced level in [GHS12].
>
> The bootstrapping for BFV is described in [CH18]. Note that BFV bootstrapping is rarely used in practice, and is not currently supported by any open-source homomorphic encryption library.
>
> The following libraries have open-source implementations of BFV (the variants are indicated in parentheses):
>
> - Microsoft SEAL (BEHZ16)
> - PALISADE (FV12, BEHZ16, HPS19) -> OpenFHE (BEHZ16, HPS19)
> - Lattigo (MTPH19) <- 읽어보니 MTPH19는 Multi-party에 관한 내용이고 base는 HPS19인듯?

**BEHZ16** [A Full RNS Variant of FV like Somewhat Homomorphic Encryption Schemes](https://eprint.iacr.org/2016/510)

**HPS19** [An Improved RNS Variant of the BFV Homomorphic Encryption Scheme](https://eprint.iacr.org/2018/117.pdf)

> Our algorithmic improvements focus on optimizing decryption and homomorphic multiplication in the Residue Number System (RNS), using
> the Chinese Remainder Theorem (CRT) to represent and manipulate
> the large coefficients in the ciphertext polynomials.

**OpenFHE**
같은 BFV scheme라 해도 어떤 mode냐에 따라 구현이 달라짐.

> OpenFHE implements four different RNS variants of the BFV scheme. These
> variants differ in the way the homomorphic multiplication is performed. There are also some differences in evaluating the decryption operation for some of the variants.
> These four variants are:
>
> - HPS:These RNS procedures use a mix of integer and floating-point operations.
> - BEHZ: These RNS procedures are based on integer arithmetic.
> - HPSPOVERQ: the HPS variant where the homomorphic encryption is optimized using the technique described in [44].
> - HPSPOVERQLEVELED: the HPSOVERQ variant where modulus switching is applied inside
>   homomorphic encryption to further reduce the computational complexity [44].
>   Note that all four methods use the modified BFV encryption method proposed in [44], which has smaller noise than the original BFV encryption method [31].
>
> OpenFHE also provides two different options for BFV encryption: STANDARD and EXTENDED. For the STANDARD option, the encryption is done using fresh modulus Q. For the EXTENDED setting, a larger modulus is used for encryption by employing auxiliary moduli available
> for homomorphic multiplication and then modulus switching to Q is executed. The EXTENDED
> option requires a slightly smaller value of Q (around 5 bits less in the case of public key encryption)
> but makes encryption more computationally expensive. The STANDARD option is used as the
> default.

어디가 얼마나 다른지 알아보아야 **하나의 스펙에 구현 디테일이 다른 것인지** 혹은 **다른 스펙의 구현체(다른 것을 구현함)인지** 판단할 수 있을 것.

## No Bootstrapping

The bootstrapping for BFV is described in [CH18]. Note that BFV bootstrapping is rarely used in practice, and **is not currently supported by any open-source homomorphic encryption library.** (from Introduction to Homomorphic Encryption and Scheme v2)

SEAL : Noise budget이라는 개념. 연산을 할 때 남은 noise budget을 계산하여, noise budget이 0bit가 되면 더 이상 decryption 했을 때 평문과 같음을 보장하지 않음.

OpenFHE : The current implementation in OpenFHE does not include noise estimation: The user specifies the
multiplicative depth (and for some schemes also the maximum number of additions/key -switching
operations), and OpenFHE selects all the necessary parameters, such as the number of bits needed
for each multiplicative level. Later, the library performs operations on the ciphertexts, applying
scaling/modulus-switching for some schemes, without trying to estimate the noise level in each
ciphertext.

## Encryption Parameters

공통적으로 N(degree of polynomials), Q(coefficient modulus of polynomials), t(modulus of plaintext)가 사용됨. 이것은 스킴에서 꼭 필요한 파라미터.

**What is P?**

- The extended ciphertext modulus. This modulus
  is used during the multiplication, and it has no impact on the security.
  See [lattigo/bfv/README.md](https://github.com/tuneinsight/lattigo/blob/master/bfv/README.md)

- It is related to hybrid key-switching.
  See this [github issue](https://github.com/tuneinsight/lattigo/issues/177) and [paper](https://eprint.iacr.org/2020/1203).

- OpenFHE에서도 KeySwitchTechnique이 HYBRID면 P가 정의됨.
  SEAL에서는 modulus chain의 last q가 P.

## Key Generation

Original BFV Scheme(Textbook scheme)ㄱ

## Example Code

### SEAL

See [SEAL/native/examples/1_bfv_basics.cpp](https://github.com/microsoft/SEAL/blob/main/native/examples/1_bfv_basics.cpp)

```cpp
    // Params, Context
    EncryptionParameters parms(scheme_type::bfv);

    // N
    size_t poly_modulus_degree = 4096;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    // Q
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
    // t
    parms.set_plain_modulus(1024);

    // Context
    // It validates the parameters for correctness, evaluates their properties,
    // and performs and stores the results of several costly pre-computations.
    // See seal/context.h
    SEALContext context(parms);

    // KeyGen
    KeyGenerator keygen(context);
    // SecretKey, PublicKey, RelinKeys, GaloisKeys 만들 수 있음
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);

    // Encyptor, Decyptor, Evaluator
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    // Plaintext
    uint64_t x = 6;
    Plaintext x_plain(uint64_to_hex_string(x));

    // Ciphertext
    Ciphertext x_encrypted;

    // Encryption
    encryptor.encrypt(x_plain, x_encrypted);

    // Decryption
    Plaintext x_decrypted;
    decryptor.decrypt(x_encrypted, x_decrypted);

    // Evaluation (square, add)
    Ciphertext x_sq_plus_one;
    evaluator.square(x_encrypted, x_sq_plus_one);
    Plaintext plain_one("1");
    evaluator.add_plain_inplace(x_sq_plus_one, plain_one);

    Plaintext decrypted_result;
    decryptor.decrypt(x_sq_plus_one, decrypted_result);

    evaluator.add_plain(x_encrypted, plain_one, x_plus_one_sq);
    evaluator.square_inplace(x_plus_one_sq);
    decryptor.decrypt(x_plus_one_sq, decrypted_result);

    Ciphertext encrypted_result;
    Plaintext plain_four("4");
    evaluator.multiply_plain_inplace(x_sq_plus_one, plain_four);
    evaluator.multiply(x_sq_plus_one, x_plus_one_sq, encrypted_result);

    // Relinearization
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);

    Ciphertext x_squared;
    evaluator.square(x_encrypted, x_squared);
    evaluator.relinearize_inplace(x_squared, relin_keys);
    evaluator.add_plain(x_squared, plain_one, x_sq_plus_one);
    decryptor.decrypt(x_sq_plus_one, decrypted_result);
    ...
```

**What is Galois keys?**

Keys for automorphism. Batching에서 plaintext를 rotate 하기 위해 사용. See [SEAL manual - 5.6 Galois Automorphisms](https://www.microsoft.com/en-us/research/uploads/prod/2017/11/sealmanual-2-3-1.pdf))

동형암호 표준 문서 : The key switching operations require the evaluator to have access to special public evaluation keys. These evaluation keys are generated by the owner of the secret key. In the context of Ciphertext-Ciphertext multiplication, these keys are often called relinearization keys; and in the context of rotation, they are sometimes called rotation or Galois keys.

### OpenFHE

See [openfhe-development/src/pke/examples/simple-integers.cpp](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/simple-integers.cpp)

```cpp
/*
  Simple example for BFVrns (integer arithmetic)
 */

#include "openfhe.h"

using namespace lbcrypto;

int main() {
    // Sample Program: Step 1: Set CryptoContext
    CCParams<CryptoContextBFVRNS> parameters;
    // t
    parameters.SetPlaintextModulus(65537);
    // Depth를 미리 설정 (LEVELEDSHE)
    parameters.SetMultiplicativeDepth(2);

    CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);
    // 의문 : N이랑 Q는?
    // -> OpenFHE에서 N은 RingDim, Q는 ciphertextModulus (include/scheme/BFVrns/cryptocontextparams-bfvrns.h)
    // 따로 주지 않으면 security를 고려하여 Default 값으로 설정 되는 듯.

    // Enable features that you wish to use
    cryptoContext->Enable(PKE);
    cryptoContext->Enable(KEYSWITCH);
    cryptoContext->Enable(LEVELEDSHE);

    // Sample Program: Step 2: Key Generation

    // Initialize Public Key Containers
    KeyPair<DCRTPoly> keyPair;
    // DCRTPoly가 뭐지?
    // - A `Poly` is a single-CRT representation using BigInteger types as coefficients, and supporting a large modulus q.
    //
    // - A `NativePoly` is a single-CRT representation using NativeInteger types, which limites the size of the coefficients and
    //   the modulus q to 64 bits.
    //
    // - A `DCRTPoly` is a double-CRT representation. In practice, this means that Poly uses a single large modulus q, while
    //   DCRTPoly uses multiple smaller moduli. Hence, Poly runs slower than DCRTPoly because DCRTPoly operations can be easier
    //   to fit into the native bitwidths of commodity processors.

    // Generate a public/private key pair
    keyPair = cryptoContext->KeyGen();

    // Generate the relinearization key
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);

    // Generate the rotation evaluation keys
    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, {1, 2, -1, -2});

    // Sample Program: Step 3: Encryption

    // First plaintext vector is encoded
    std::vector<int64_t> vectorOfInts1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Plaintext plaintext1               = cryptoContext->MakePackedPlaintext(vectorOfInts1);
    // Second plaintext vector is encoded
    std::vector<int64_t> vectorOfInts2 = {3, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Plaintext plaintext2               = cryptoContext->MakePackedPlaintext(vectorOfInts2);
    // Third plaintext vector is encoded
    std::vector<int64_t> vectorOfInts3 = {1, 2, 5, 2, 5, 6, 7, 8, 9, 10, 11, 12};
    Plaintext plaintext3               = cryptoContext->MakePackedPlaintext(vectorOfInts3);

    // The encoded vectors are encrypted
    auto ciphertext1 = cryptoContext->Encrypt(keyPair.publicKey, plaintext1);
    auto ciphertext2 = cryptoContext->Encrypt(keyPair.publicKey, plaintext2);
    auto ciphertext3 = cryptoContext->Encrypt(keyPair.publicKey, plaintext3);

    // Sample Program: Step 4: Evaluation

    // Homomorphic additions
    auto ciphertextAdd12     = cryptoContext->EvalAdd(ciphertext1, ciphertext2);
    auto ciphertextAddResult = cryptoContext->EvalAdd(ciphertextAdd12, ciphertext3);

    // Homomorphic multiplications
    auto ciphertextMul12      = cryptoContext->EvalMult(ciphertext1, ciphertext2);
    auto ciphertextMultResult = cryptoContext->EvalMult(ciphertextMul12, ciphertext3);

    // Homomorphic rotations
    auto ciphertextRot1 = cryptoContext->EvalRotate(ciphertext1, 1);
    auto ciphertextRot2 = cryptoContext->EvalRotate(ciphertext1, 2);
    auto ciphertextRot3 = cryptoContext->EvalRotate(ciphertext1, -1);
    auto ciphertextRot4 = cryptoContext->EvalRotate(ciphertext1, -2);

    // Sample Program: Step 5: Decryption

    // Decrypt the result of additions
    Plaintext plaintextAddResult;
    cryptoContext->Decrypt(keyPair.secretKey, ciphertextAddResult, &plaintextAddResult);

    // Decrypt the result of multiplications
    Plaintext plaintextMultResult;
    cryptoContext->Decrypt(keyPair.secretKey, ciphertextMultResult, &plaintextMultResult);

    // Decrypt the result of rotations
    Plaintext plaintextRot1;
    cryptoContext->Decrypt(keyPair.secretKey, ciphertextRot1, &plaintextRot1);
    Plaintext plaintextRot2;
    cryptoContext->Decrypt(keyPair.secretKey, ciphertextRot2, &plaintextRot2);
    Plaintext plaintextRot3;
    cryptoContext->Decrypt(keyPair.secretKey, ciphertextRot3, &plaintextRot3);
    Plaintext plaintextRot4;
    cryptoContext->Decrypt(keyPair.secretKey, ciphertextRot4, &plaintextRot4);

    plaintextRot1->SetLength(vectorOfInts1.size());
    plaintextRot2->SetLength(vectorOfInts1.size());
    plaintextRot3->SetLength(vectorOfInts1.size());
    plaintextRot4->SetLength(vectorOfInts1.size());

    std::cout << "Plaintext #1: " << plaintext1 << std::endl;
    std::cout << "Plaintext #2: " << plaintext2 << std::endl;
    std::cout << "Plaintext #3: " << plaintext3 << std::endl;

    // Output results
    std::cout << "\nResults of homomorphic computations" << std::endl;
    std::cout << "#1 + #2 + #3: " << plaintextAddResult << std::endl;
    std::cout << "#1 * #2 * #3: " << plaintextMultResult << std::endl;
    std::cout << "Left rotation of #1 by 1: " << plaintextRot1 << std::endl;
    std::cout << "Left rotation of #1 by 2: " << plaintextRot2 << std::endl;
    std::cout << "Right rotation of #1 by 1: " << plaintextRot3 << std::endl;
    std::cout << "Right rotation of #1 by 2: " << plaintextRot4 << std::endl;

    return 0;
}
```

### Lattigo

See [lattigo/examples/bfv/main.go](https://github.com/tuneinsight/lattigo/blob/master/examples/bfv/main.go)

NOTE : PN13QP218 is a set of default parameters with logN=13 and logQP=218

```go
// BFV parameters (128 bit security) with plaintext modulus 65929217
paramDef := bfv.PN13QP218 // N,Q,P,T
//t
paramDef.T = 0x3ee0001 //65929217

params, err := bfv.NewParametersFromLiteral(paramDef)
if err != nil {
 panic(err)
}

encoder := bfv.NewEncoder(params)

// Rider's keygen
kgen := bfv.NewKeyGenerator(params)

riderSk, riderPk := kgen.GenKeyPair()

decryptor := bfv.NewDecryptor(params, riderSk)

encryptorRiderPk := bfv.NewEncryptor(params, riderPk)

encryptorRiderSk := bfv.NewEncryptor(params, riderSk)

evaluator := bfv.NewEvaluator(params, rlwe.EvaluationKey{})

fmt.Printf("Parameters : N=%d, T=%d, Q = %d bits, sigma = %f \n",
 1<<params.LogN(), params.T(), params.LogQP(), params.Sigma())
fmt.Println()

maxvalue := uint64(math.Sqrt(float64(params.T()))) // max values = floor(sqrt(plaintext modulus))
mask := uint64(1<<bits.Len64(maxvalue) - 1)        // binary mask upper-bound for the uniform sampling

prng, err := utils.NewPRNG()
if err != nil {
 panic(err)
}
// Rider coordinates [x, y, x, y, ....., x, y]
riderPosX, riderPosY := ring.RandUniform(prng, maxvalue, mask), ring.RandUniform(prng, maxvalue, mask)

Rider := make([]uint64, 1<<params.LogN())
for i := 0; i < nbDrivers; i++ {
 Rider[(i << 1)] = riderPosX
 Rider[(i<<1)+1] = riderPosY
}

riderPlaintext := bfv.NewPlaintext(params, params.MaxLevel())
encoder.Encode(Rider, riderPlaintext)

// driversData coordinates [0, 0, ..., x, y, ..., 0, 0]
driversData := make([][]uint64, nbDrivers)

driversPlaintexts := make([]*rlwe.Plaintext, nbDrivers)
for i := 0; i < nbDrivers; i++ {
 driversData[i] = make([]uint64, 1<<params.LogN())
 driversData[i][(i << 1)] = ring.RandUniform(prng, maxvalue, mask)
 driversData[i][(i<<1)+1] = ring.RandUniform(prng, maxvalue, mask)
 driversPlaintexts[i] = bfv.NewPlaintext(params, params.MaxLevel())
 encoder.Encode(driversData[i], driversPlaintexts[i])
}

RiderCiphertext := encryptorRiderSk.EncryptNew(riderPlaintext)

DriversCiphertexts := make([]*rlwe.Ciphertext, nbDrivers)
for i := 0; i < nbDrivers; i++ {
 DriversCiphertexts[i] = encryptorRiderPk.EncryptNew(driversPlaintexts[i])
}

evaluator.Neg(RiderCiphertext, RiderCiphertext)
for i := 0; i < nbDrivers; i++ {
 evaluator.Add(RiderCiphertext, DriversCiphertexts[i], RiderCiphertext)
}

result := encoder.DecodeUintNew(decryptor.DecryptNew(evaluator.MulNew(RiderCiphertext, RiderCiphertext)))

minIndex, minPosX, minPosY, minDist := 0, params.T(), params.T(), params.T()

errors := 0

for i := 0; i < nbDrivers; i++ {

 driverPosX, driverPosY := driversData[i][i<<1], driversData[i][(i<<1)+1]

 computedDist := result[i<<1] + result[(i<<1)+1]
 expectedDist := distance(driverPosX, driverPosY, riderPosX, riderPosY)

 if computedDist == expectedDist {
  if computedDist < minDist {
   minIndex = i
   minPosX, minPosY = driverPosX, driverPosY
   minDist = computedDist
  }
 } else {
  errors++
 }

 if i < 4 || i > nbDrivers-5 {
   i, computedDist, driverPosX, riderPosX, driverPosY, riderPosY, computedDist == expectedDist)
 }

}

```

## Tests

### SEAL

#### EncryptionParametersTest

See [native/tests/seal/encryptionparams.cpp](https://github.com/microsoft/SEAL/blob/206648d0e4634e5c61dcf9370676630268290b59/native/tests/seal/encryptionparams.cpp)

##### EncryptionParametersSet

- params가 잘 정의 되는지 테스트.
- coeff modulus가 소수인지도 확인

##### EncryptionParametersCompare

- params equality test

#### EncryptionParametersSaveLoad

- params를 저장하고 불러오는 기능 테스트.

#### ContextTest

See [native/tests/seal/context.cpp](https://github.com/microsoft/SEAL/blob/206648d0e4634e5c61dcf9370676630268290b59/native/tests/seal/context.cpp)

##### BFVContextConstructor

- 파라미터 N, Q, t와 expand_mod_chain(bool),sec_level_type 값에 따라 context의 property(context 내에서 fft,ntt,batching,keyswitching 등이 가능한지? 자세한 내용은 seal/native/src/seal/context.h 참조) 테스트.

##### ModulusChainExpansion

- expand_mod_chain == true 일 때, context 내에서 modulus chain이 잘 정의되었는지 확인. 이 테스트에서 "잘 정의됨"을 확인하는 oracle이 무엇인지? document에서는 안보임. 이런 부분을 spec에 담아야하지 않나?

-> 확인해보니 쉽게 유추됨. total_coeff_modulus = modulus chain에 있는 전체 modulus의 곱 등. 그러나 여전히 modulus chain의 정의를 specify할 필요가 있지 않나 생각됨.

##### BFVParameterError

- Parameter error msg test

### OpenFHE

- TODO

### Lattigo

- TODO
