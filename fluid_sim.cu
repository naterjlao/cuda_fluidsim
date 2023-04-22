void jacobi(half2 coords   : WPOS,   // grid coordinates     out    half4 xNew : COLOR,  // result     uniform    half alpha,             uniform    half rBeta,      // reciprocal beta     uniform samplerRECT x,   // x vector (Ax = b)     uniform samplerRECT b)   // b vector (Ax = b) {   // left, right, bottom, and top x samples    half4 xL = h4texRECT(x, coords - half2(1, 0));   half4 xR = h4texRECT(x, coords + half2(1, 0));
half4 xB = h4texRECT(x, coords - half2(0, 1));   half4 xT = h4texRECT(x, coords + half2(0, 1)); 
  // b sample, from center
  half4 bC = h4texRECT(b, coords); 
  // evaluate Jacobi iteration
  xNew = (xL + xR + xB + xT + alpha * bC) * rBeta; }