{- |
Raw, unnormalized versions of the transforms in @fftw@.

Note that the forwards and backwards transforms of this module are not actually
inverses.  For example, @run idft (run dft v) /= v@ in general.

For more information on the individual transforms, see
<http://www.fftw.org/fftw3_doc/What-FFTW-Really-Computes.html>.
-}
module Numeric.FFT.Vector.Unnormalized(
                    -- * Creating and executing 'Plan's
                    run,
                    plan,
                    execute,
                    -- * Complex-to-complex transforms
                    -- ** One-dimensional complex-to-complex transforms
                    dft,
                    idft,
                    -- ** Two-dimensional complex-to-complex transforms
                    dft2d,
                    idft2d,
                    -- ** Three-dimensional complex-to-complex transforms
                    dft3d,
                    idft3d,
                    -- * Real-to-complex transforms
                    -- ** One-dimensional real-to-complex transforms
                    dftR2C,
                    dftC2R,
                    -- ** Two-dimensional real-to-complex transforms
                    dftR2C2D,
                    dftC2R2D,
                    -- ** Three-dimensional real-to-complex transforms
                    dftR2C3D,
                    dftC2R3D,
                    -- * Real-to-real transforms
                    -- $dct_size

                    -- ** Discrete cosine transforms
                    -- *** One-dimensional discrete cosine transforms
                    dct1,
                    dct2,
                    dct3,
                    dct4,
                    -- *** Two-dimensional discrete cosine transforms
                    dct1_2D,
                    dct2_2D,
                    dct3_2D,
                    dct4_2D,
                    -- *** Three-dimensional discrete cosine transforms
                    dct1_3D,
                    dct2_3D,
                    dct3_3D,
                    dct4_3D,
                    -- ** Discrete sine transforms
                    -- *** One-dimensional discrete sine transforms
                    dst1,
                    dst2,
                    dst3,
                    dst4,
                    -- *** Two-dimensional discrete sine transforms
                    dst1_2D,
                    dst2_2D,
                    dst3_2D,
                    dst4_2D,
                    -- *** Three-dimensional discrete sine transforms
                    dst1_3D,
                    dst2_3D,
                    dst3_3D,
                    dst4_3D,
                    ) where

import Numeric.FFT.Vector.Base
import Foreign
import Foreign.C
import Data.Complex
import qualified Data.List as List ( intersperse )

-- $setup
-- >>> import qualified Data.Vector.Storable as VS
-- >>> import Data.Vector.Storable ( Vector )

#include <fftw3.h>

-- | Whether the complex fft is forwards or backwards.
type CDirection = CInt

-- | The type of the cosine or sine transform.
type CKind = (#type fftw_r2r_kind)

foreign import ccall unsafe fftw_plan_dft_1d
    :: CInt -> Ptr (Complex Double) -> Ptr (Complex Double) -> CDirection
        -> CFlags -> IO (Ptr CPlan)

foreign import ccall unsafe fftw_plan_dft_2d
    :: CInt -> CInt -> Ptr (Complex Double) -> Ptr (Complex Double)
        -> CDirection -> CFlags -> IO (Ptr CPlan)

foreign import ccall unsafe fftw_plan_dft_3d
    :: CInt -> CInt -> CInt -> Ptr (Complex Double) -> Ptr (Complex Double)
        -> CDirection -> CFlags -> IO (Ptr CPlan)

foreign import ccall unsafe fftw_plan_dft_r2c_1d
    :: CInt -> Ptr Double -> Ptr (Complex Double) -> CFlags -> IO (Ptr CPlan)

foreign import ccall unsafe fftw_plan_dft_c2r_1d
    :: CInt -> Ptr (Complex Double) -> Ptr Double -> CFlags -> IO (Ptr CPlan)

foreign import ccall unsafe fftw_plan_dft_r2c_2d
    :: CInt -> CInt -> Ptr Double -> Ptr (Complex Double)
        -> CFlags -> IO (Ptr CPlan)

foreign import ccall unsafe fftw_plan_dft_c2r_2d
    :: CInt -> CInt -> Ptr (Complex Double) -> Ptr Double
        -> CFlags -> IO (Ptr CPlan)

foreign import ccall unsafe fftw_plan_dft_r2c_3d
    :: CInt -> CInt -> CInt -> Ptr Double -> Ptr (Complex Double)
        -> CFlags -> IO (Ptr CPlan)

foreign import ccall unsafe fftw_plan_dft_c2r_3d
    :: CInt -> CInt -> CInt -> Ptr (Complex Double) -> Ptr Double
        -> CFlags -> IO (Ptr CPlan)

foreign import ccall unsafe fftw_plan_r2r_1d
    :: CInt -> Ptr Double -> Ptr Double -> CKind -> CFlags -> IO (Ptr CPlan)

foreign import ccall unsafe fftw_plan_r2r_2d
    :: CInt -> CInt -> Ptr Double -> Ptr Double
        -> CKind -> CFlags -> IO (Ptr CPlan)

foreign import ccall unsafe fftw_plan_r2r_3d
    :: CInt -> CInt -> CInt -> Ptr Double -> Ptr Double
        -> CKind -> CFlags -> IO (Ptr CPlan)

dft1D :: CDirection -> Transform (Complex Double) (Complex Double)
dft1D d = Transform {
            inputSize = id,
            outputSize = id,
            creationSizeFromInput = id,
            makePlan = \n a b -> withPlanner . fftw_plan_dft_1d n a b d,
            normalization = const id
            }

-- | A forward discrete Fourier transform.  The output and input sizes are the same (@n@).
--
-- @y_k = sum_(j=0)^(n-1) x_j e^(-2pi i j k/n)@
dft :: Transform (Complex Double) (Complex Double)
dft = dft1D (#const FFTW_FORWARD)

-- | A backward discrete Fourier transform.  The output and input sizes are the same (@n@).
--
-- @y_k = sum_(j=0)^(n-1) x_j e^(2pi i j k/n)@
idft :: Transform (Complex Double) (Complex Double)
idft = dft1D (#const FFTW_BACKWARD)

-- | A forward discrete Fourier transform with real data.  If the input size is @n@,
-- the output size will be @n \`div\` 2 + 1@.
dftR2C :: Transform Double (Complex Double)
dftR2C = Transform {
            inputSize = id,
            outputSize = \n -> n `div` 2 + 1,
            creationSizeFromInput = id,
            makePlan = \n a b -> withPlanner . fftw_plan_dft_r2c_1d n a b,
            normalization = const id
        }

-- | A backward discrete Fourier transform which produces real data.
--
-- This 'Transform' behaves differently than the others:
--
--  - Calling @plan dftC2R n@ creates a 'Plan' whose /output/ size is @n@, and whose
--    /input/ size is @n \`div\` 2 + 1@.
--
--  - If @length v == n@, then @length (run dftC2R v) == 2*(n-1)@.
dftC2R :: Transform (Complex Double) Double
dftC2R = Transform {
            inputSize = \n -> n `div` 2 + 1,
            outputSize = id,
            creationSizeFromInput = \n -> 2 * (n-1),
            makePlan = \n a b -> withPlanner . fftw_plan_dft_c2r_1d n a b,
            normalization = const id
        }

r2rTransform :: CKind -> Transform Double Double
r2rTransform kind = Transform {
                    inputSize = id,
                    outputSize = id,
                    creationSizeFromInput = id,
                    makePlan = \n a b -> withPlanner . fftw_plan_r2r_1d n a b kind,
                    normalization = const id
                }

-- $dct_size
-- The real-even (DCT) and real-odd (DST) transforms.  The input and output sizes
-- are the same (@n@).

-- | A type-1 discrete cosine transform.
--
-- @y_k = x_0 + (-1)^k x_(n-1) + 2 sum_(j=1)^(n-2) x_j cos(pi j k\/(n-1))@
dct1 :: Transform Double Double
dct1 = r2rTransform (#const  FFTW_REDFT00)

-- | A type-2 discrete cosine transform.
--
-- @y_k = 2 sum_(j=0)^(n-1) x_j cos(pi(j+1\/2)k\/n)@
dct2 :: Transform Double Double
dct2 = r2rTransform (#const  FFTW_REDFT10)

-- | A type-3 discrete cosine transform.
--
-- @y_k = x_0 + 2 sum_(j=1)^(n-1) x_j cos(pi j(k+1\/2)\/n)@
dct3 :: Transform Double Double
dct3 = r2rTransform (#const  FFTW_REDFT01)

-- | A type-4 discrete cosine transform.
--
-- @y_k = 2 sum_(j=0)^(n-1) x_j cos(pi(j+1\/2)(k+1\/2)\/n)@
dct4 :: Transform Double Double
dct4 = r2rTransform (#const  FFTW_REDFT11)

-- | A type-1 discrete sine transform.
--
-- @y_k = 2 sum_(j=0)^(n-1) x_j sin(pi(j+1)(k+1)\/(n+1))@
dst1 :: Transform Double Double
dst1 = r2rTransform (#const  FFTW_RODFT00)

-- | A type-2 discrete sine transform.
--
-- @y_k = 2 sum_(j=0)^(n-1) x_j sin(pi(j+1\/2)(k+1)\/n)@
dst2 :: Transform Double Double
dst2 = r2rTransform (#const  FFTW_RODFT10)

-- | A type-3 discrete sine transform.
--
-- @y_k = (-1)^k x_(n-1) + 2 sum_(j=0)^(n-2) x_j sin(pi(j+1)(k+1\/2)/n)@
dst3 :: Transform Double Double
dst3 = r2rTransform (#const  FFTW_RODFT01)

-- | A type-4 discrete sine transform.
--
-- @y_k = sum_(j=0)^(n-1) x_j sin(pi(j+1\/2)(k+1\/2)\/n)@
dst4 :: Transform Double Double
dst4 = r2rTransform (#const FFTW_RODFT11)

{- Multi-dimensional transforms -}

checkSize :: (Show a, Integral a) => a -> [a] -> String -> b -> b
checkSize n lst funcname f =
  if n /= product lst
  then error $
       "Numeric.FFT.Vector.Unnormalized." ++ funcname ++
       ": size mismatch: " ++ show n ++ " /= " ++
       (concat . List.intersperse " * " . map show $ lst)
  else f

dft2D :: CDirection -> (Int, Int) -> Transform (Complex Double) (Complex Double)
dft2D d (rows, cols) = Transform {
  inputSize = id,
  outputSize = id,
  creationSizeFromInput = id,
  makePlan = \n a b -> checkSize n [r, c] "dft2D" $
                       withPlanner . fftw_plan_dft_2d r c a b d,
  normalization = const id
  }
  where
    r = fromIntegral rows
    c = fromIntegral cols

-- | A forward discrete Fourier transform with real data.  If the
-- input size is @n = rows * cols@, the output size will be @n \`div\`
-- rows \`div\` 2 + 1@.
dftR2C2D :: (Int, Int) -> Transform Double (Complex Double)
dftR2C2D (rows, cols) = Transform {
  inputSize = id,
  outputSize = \n -> rows * (n `div` rows `div` 2 + 1),
  creationSizeFromInput = id,
  makePlan = \n a b -> checkSize n [r, c] "dftR2C2D" $
                       withPlanner . fftw_plan_dft_r2c_2d r c a b,
  normalization = const id
  }
  where
    r = fromIntegral rows
    c = fromIntegral cols

-- | A backward discrete Fourier transform which produces real data.
--
-- This 'Transform' behaves differently than the others:
--
--  - Calling @plan dftC2R n@ creates a 'Plan' whose /output/ size is
--    @n = rows * cols@, and whose /input/ size is @n \`div\` rows \`div\` 2 + 1@.
--
--  - If @length v == n = rows * cols@, then @length (run dftC2R v) ==
--    rows * 2 * (cols - 1)@.
dftC2R2D :: (Int, Int) -> Transform (Complex Double) Double
dftC2R2D (rows, cols) = Transform {
  inputSize = \n -> rows * expandCols n,
  outputSize = id,
  creationSizeFromInput = \n -> rows * 2 * (n `div` rows - 1),
  makePlan = \n a b -> checkSize (r * expandCols n) [r, c] "dftC2R2D" $
                       withPlanner . fftw_plan_dft_c2r_2d r c a b,
  normalization = const id
  }
  where
    r = fromIntegral rows
    c = fromIntegral cols
    expandCols n = n `div` fromIntegral rows `div` 2 + 1

-- | A forward discrete Fourier transform.  The output and input sizes are the same (@n@).
--
-- @y_k = sum_(j=0)^(n-1) x_j e^(-2pi i j k/n)@
dft2d :: (Int, Int) -> Transform (Complex Double) (Complex Double)
dft2d = dft2D (#const FFTW_FORWARD)

-- | A backward discrete Fourier transform.  The output and input sizes are the same (@n@).
--
-- @y_k = sum_(j=0)^(n-1) x_j e^(2pi i j k/n)@
idft2d :: (Int, Int) -> Transform (Complex Double) (Complex Double)
idft2d = dft2D (#const FFTW_BACKWARD)

r2rTransform2D :: CKind -> (Int, Int) -> Transform Double Double
r2rTransform2D kind (rows, cols) = Transform {
  inputSize = id,
  outputSize = id,
  creationSizeFromInput = id,
  makePlan = \n a b -> checkSize n [r, c] "r2rTransform2D" $
                       withPlanner . fftw_plan_r2r_2d r c a b kind,
  normalization = const id
  }
  where
    r = fromIntegral rows
    c = fromIntegral cols

-- | A type-1 discrete cosine transform.
--
-- @y_k = x_0 + (-1)^k x_(n-1) + 2 sum_(j=1)^(n-2) x_j cos(pi j k\/(n-1))@
dct1_2D :: (Int, Int) -> Transform Double Double
dct1_2D = r2rTransform2D (#const  FFTW_REDFT00)

-- | A type-2 discrete cosine transform.
--
-- @y_k = 2 sum_(j=0)^(n-1) x_j cos(pi(j+1\/2)k\/n)@
dct2_2D :: (Int, Int) -> Transform Double Double
dct2_2D = r2rTransform2D (#const  FFTW_REDFT10)

-- | A type-3 discrete cosine transform.
--
-- @y_k = x_0 + 2 sum_(j=1)^(n-1) x_j cos(pi j(k+1\/2)\/n)@
dct3_2D :: (Int, Int) -> Transform Double Double
dct3_2D = r2rTransform2D (#const  FFTW_REDFT01)

-- | A type-4 discrete cosine transform.
--
-- @y_k = 2 sum_(j=0)^(n-1) x_j cos(pi(j+1\/2)(k+1\/2)\/n)@
dct4_2D :: (Int, Int) -> Transform Double Double
dct4_2D = r2rTransform2D (#const  FFTW_REDFT11)

-- | A type-1 discrete sine transform.
--
-- @y_k = 2 sum_(j=0)^(n-1) x_j sin(pi(j+1)(k+1)\/(n+1))@
dst1_2D :: (Int, Int) -> Transform Double Double
dst1_2D = r2rTransform2D (#const  FFTW_RODFT00)

-- | A type-2 discrete sine transform.
--
-- @y_k = 2 sum_(j=0)^(n-1) x_j sin(pi(j+1\/2)(k+1)\/n)@
dst2_2D :: (Int, Int) -> Transform Double Double
dst2_2D = r2rTransform2D (#const  FFTW_RODFT10)

-- | A type-3 discrete sine transform.
--
-- @y_k = (-1)^k x_(n-1) + 2 sum_(j=0)^(n-2) x_j sin(pi(j+1)(k+1\/2)/n)@
dst3_2D :: (Int, Int) -> Transform Double Double
dst3_2D = r2rTransform2D (#const  FFTW_RODFT01)

-- | A type-4 discrete sine transform.
--
-- @y_k = sum_(j=0)^(n-1) x_j sin(pi(j+1\/2)(k+1\/2)\/n)@
dst4_2D :: (Int, Int) -> Transform Double Double
dst4_2D = r2rTransform2D (#const FFTW_RODFT11)

dft3D :: CDirection -> (Int, Int, Int) -> Transform (Complex Double) (Complex Double)
dft3D d (rows, cols, layers) = Transform {
  inputSize = id,
  outputSize = id,
  creationSizeFromInput = id,
  makePlan = \n a b -> checkSize n [r, c, l] "dft3D" $
                       withPlanner . fftw_plan_dft_3d r c l a b d,
  normalization = const id
  }
  where
    r = fromIntegral rows
    c = fromIntegral cols
    l = fromIntegral layers

-- | A forward discrete Fourier transform.  The output and input sizes are the same (@n@).
--
-- @y_k = sum_(j=0)^(n-1) x_j e^(-2pi i j k/n)@
dft3d :: (Int, Int, Int) -> Transform (Complex Double) (Complex Double)
dft3d = dft3D (#const FFTW_FORWARD)

-- | A backward discrete Fourier transform.  The output and input sizes are the same (@n@).
--
-- @y_k = sum_(j=0)^(n-1) x_j e^(2pi i j k/n)@
idft3d :: (Int, Int, Int) -> Transform (Complex Double) (Complex Double)
idft3d = dft3D (#const FFTW_BACKWARD)

-- | A forward discrete Fourier transform with real data.  If the
-- input size is @n = rows * cols * layers@, the output size will be
-- @n \`div\` rows \`div\` cols \`div\` 2 + 1@.
dftR2C3D :: (Int, Int, Int) -> Transform Double (Complex Double)
dftR2C3D (rows, cols, layers) = Transform {
  inputSize = id,
  outputSize = \n -> n `div` rows `div` cols `div` 2 + 1,
  creationSizeFromInput = id,
  makePlan = \n a b -> checkSize n [r, c, l] "dftR2C3D" $
                       withPlanner . fftw_plan_dft_r2c_3d r c l a b,
  normalization = const id
  }
  where
    r = fromIntegral rows
    c = fromIntegral cols
    l = fromIntegral layers

-- | A backward discrete Fourier transform which produces real data.
--
-- This 'Transform' behaves differently than the others:
--
--  - Calling @plan dftC2R (rows, cols, layers)@ creates a 'Plan'
--    whose /output/ size is @n = rows * cols * layers@, and whose
--    /input/ size is @n \`div\` rows \`div\` cols \`div\` 2 + 1@.
--
--  - If @length v == n = rows * cols * layers@, then @length (run
--    dftC2R v) == rows * cols * 2 * (layers - 1)@.
dftC2R3D :: (Int, Int, Int) -> Transform (Complex Double) Double
dftC2R3D (rows, cols, layers) = Transform {
  inputSize = \n -> rows * cols * expandLayers n,
  outputSize = id,
  creationSizeFromInput = \n -> rows * cols * 2 * (n `div` rows `div` cols - 1),
  makePlan = \n a b -> checkSize (r * c * expandLayers n) [r, c, l] "dftC2R3D" $
                       withPlanner . fftw_plan_dft_c2r_3d r c l a b,
  normalization = const id
  }
  where
    r = fromIntegral rows
    c = fromIntegral cols
    l = fromIntegral layers
    expandLayers n =
      n `div` fromIntegral rows `div` fromIntegral cols `div` 2 + 1

r2rTransform3D :: CKind -> (Int, Int, Int) -> Transform Double Double
r2rTransform3D kind (rows, cols, layers) = Transform {
  inputSize = id,
  outputSize = id,
  creationSizeFromInput = id,
  makePlan = \n a b -> checkSize n [r, c, l] "r2rTransform3D" $
                       withPlanner . fftw_plan_r2r_3d r c l a b kind,
  normalization = const id
  }
  where
    r = fromIntegral rows
    c = fromIntegral cols
    l = fromIntegral layers

-- | A type-1 discrete cosine transform.
--
-- @y_k = x_0 + (-1)^k x_(n-1) + 2 sum_(j=1)^(n-2) x_j cos(pi j k\/(n-1))@
dct1_3D :: (Int, Int, Int) -> Transform Double Double
dct1_3D = r2rTransform3D (#const  FFTW_REDFT00)

-- | A type-2 discrete cosine transform.
--
-- @y_k = 2 sum_(j=0)^(n-1) x_j cos(pi(j+1\/2)k\/n)@
dct2_3D :: (Int, Int, Int) -> Transform Double Double
dct2_3D = r2rTransform3D (#const  FFTW_REDFT10)

-- | A type-3 discrete cosine transform.
--
-- @y_k = x_0 + 2 sum_(j=1)^(n-1) x_j cos(pi j(k+1\/2)\/n)@
dct3_3D :: (Int, Int, Int) -> Transform Double Double
dct3_3D = r2rTransform3D (#const  FFTW_REDFT01)

-- | A type-4 discrete cosine transform.
--
-- @y_k = 2 sum_(j=0)^(n-1) x_j cos(pi(j+1\/2)(k+1\/2)\/n)@
dct4_3D :: (Int, Int, Int) -> Transform Double Double
dct4_3D = r2rTransform3D (#const  FFTW_REDFT11)

-- | A type-1 discrete sine transform.
--
-- @y_k = 2 sum_(j=0)^(n-1) x_j sin(pi(j+1)(k+1)\/(n+1))@
dst1_3D :: (Int, Int, Int) -> Transform Double Double
dst1_3D = r2rTransform3D (#const  FFTW_RODFT00)

-- | A type-2 discrete sine transform.
--
-- @y_k = 2 sum_(j=0)^(n-1) x_j sin(pi(j+1\/2)(k+1)\/n)@
dst2_3D :: (Int, Int, Int) -> Transform Double Double
dst2_3D = r2rTransform3D (#const  FFTW_RODFT10)

-- | A type-3 discrete sine transform.
--
-- @y_k = (-1)^k x_(n-1) + 2 sum_(j=0)^(n-2) x_j sin(pi(j+1)(k+1\/2)/n)@
dst3_3D :: (Int, Int, Int) -> Transform Double Double
dst3_3D = r2rTransform3D (#const  FFTW_RODFT01)

-- | A type-4 discrete sine transform.
--
-- @y_k = sum_(j=0)^(n-1) x_j sin(pi(j+1\/2)(k+1\/2)\/n)@
dst4_3D :: (Int, Int, Int) -> Transform Double Double
dst4_3D = r2rTransform3D (#const FFTW_RODFT11)
