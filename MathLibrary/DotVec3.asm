; ============================================================================
; DotVec3.asm  -  MASM x64 (Visual Studio : clic droit projet > Build Dependencies >
;                 Build Customizations > cocher "masm (.targets, .props)")
; Convention Windows x64 : args entiers rcx, rdx, r8, r9 ; retour float dans xmm0.
; xmm0-xmm5 sont "volatiles" : on peut les écraser sans les sauvegarder.
; Ordre d'addition (x + y) + z : identique à la référence C++ -> résultat identique bit à bit.
; ============================================================================
.code

; float dot_vec3_asm(const float* a /*rcx*/, const float* b /*rdx*/)
dot_vec3_asm PROC
    movups  xmm0, xmmword ptr [rcx]     ; xmm0 = a.x a.y a.z a.w
    movups  xmm1, xmmword ptr [rdx]     ; xmm1 = b.x b.y b.z b.w
    mulps   xmm0, xmm1                  ; xmm0 = ax*bx  ay*by  az*bz  aw*bw   (lane w ignorée)
    movaps  xmm1, xmm0
    shufps  xmm1, xmm1, 055h            ; _MM_SHUFFLE(1,1,1,1) : y*y' dans la lane 0
    movaps  xmm2, xmm0
    shufps  xmm2, xmm2, 0AAh            ; _MM_SHUFFLE(2,2,2,2) : z*z' dans la lane 0
    addss   xmm0, xmm1                  ; lane 0 = x*x' + y*y'
    addss   xmm0, xmm2                  ; lane 0 = (x*x' + y*y') + z*z'
    ret                                 ; résultat = lane 0 de xmm0
dot_vec3_asm ENDP

; void dot_batch_asm(const Vector3* a /*rcx*/, const Vector3* b /*rdx*/, float* out /*r8*/, size_t n /*r9*/)
dot_batch_asm PROC
    test    r9, r9                      ; n == 0 ? rien n'est lu ni écrit
    jz      batch_done
batch_loop:
    movups  xmm0, xmmword ptr [rcx]
    movups  xmm1, xmmword ptr [rdx]
    mulps   xmm0, xmm1
    movaps  xmm1, xmm0
    shufps  xmm1, xmm1, 055h
    movaps  xmm2, xmm0
    shufps  xmm2, xmm2, 0AAh
    addss   xmm0, xmm1
    addss   xmm0, xmm2
    movss   dword ptr [r8], xmm0        ; out[i] = dot
    add     rcx, 16                     ; sizeof(Vector3) = 16
    add     rdx, 16
    add     r8, 4                       ; sizeof(float)
    dec     r9
    jnz     batch_loop
batch_done:
    ret
dot_batch_asm ENDP

END