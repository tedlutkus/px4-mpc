
/*
Auto-generated by CVXPYgen on November 18, 2024 at 14:09:49.
Content: Example program for updating parameters, solving, and inspecting the result.
*/

#include <stdio.h>
#include "cpg_workspace.h"
#include "cpg_solve.h"

static int i;

int main(int argc, char *argv[]){

  // Update first entry of every user-defined parameter
  cpg_update_param5(0, -0.36221447060863976031);
  cpg_update_param1(0, 0.26292005540569884925);
  cpg_update_param2(0, 0.10069326875981828018);

  // Solve the problem instance
  cpg_solve();

  // Print objective function value
  printf("obj = %f\n", CPG_Result.info->obj_val);

  // Print primal solution
  for(i=0; i<143; i++) {
    printf("var4[%d] = %f\n", i, CPG_Result.prim->var4[i]);
  }
  for(i=0; i<40; i++) {
    printf("var3[%d] = %f\n", i, CPG_Result.prim->var3[i]);
  }

  // Print dual solution
  for(i=0; i<13; i++) {
    printf("d0[%d] = %f\n", i, CPG_Result.dual->d0[i]);
  }
  for(i=0; i<13; i++) {
    printf("d1[%d] = %f\n", i, CPG_Result.dual->d1[i]);
  }
  for(i=0; i<4; i++) {
    printf("d2[%d] = %f\n", i, CPG_Result.dual->d2[i]);
  }
  for(i=0; i<4; i++) {
    printf("d3[%d] = %f\n", i, CPG_Result.dual->d3[i]);
  }
  for(i=0; i<13; i++) {
    printf("d4[%d] = %f\n", i, CPG_Result.dual->d4[i]);
  }
  for(i=0; i<4; i++) {
    printf("d5[%d] = %f\n", i, CPG_Result.dual->d5[i]);
  }
  for(i=0; i<4; i++) {
    printf("d6[%d] = %f\n", i, CPG_Result.dual->d6[i]);
  }
  for(i=0; i<13; i++) {
    printf("d7[%d] = %f\n", i, CPG_Result.dual->d7[i]);
  }
  for(i=0; i<4; i++) {
    printf("d8[%d] = %f\n", i, CPG_Result.dual->d8[i]);
  }
  for(i=0; i<4; i++) {
    printf("d9[%d] = %f\n", i, CPG_Result.dual->d9[i]);
  }
  for(i=0; i<13; i++) {
    printf("d10[%d] = %f\n", i, CPG_Result.dual->d10[i]);
  }
  for(i=0; i<4; i++) {
    printf("d11[%d] = %f\n", i, CPG_Result.dual->d11[i]);
  }
  for(i=0; i<4; i++) {
    printf("d12[%d] = %f\n", i, CPG_Result.dual->d12[i]);
  }
  for(i=0; i<13; i++) {
    printf("d13[%d] = %f\n", i, CPG_Result.dual->d13[i]);
  }
  for(i=0; i<4; i++) {
    printf("d14[%d] = %f\n", i, CPG_Result.dual->d14[i]);
  }
  for(i=0; i<4; i++) {
    printf("d15[%d] = %f\n", i, CPG_Result.dual->d15[i]);
  }
  for(i=0; i<13; i++) {
    printf("d16[%d] = %f\n", i, CPG_Result.dual->d16[i]);
  }
  for(i=0; i<4; i++) {
    printf("d17[%d] = %f\n", i, CPG_Result.dual->d17[i]);
  }
  for(i=0; i<4; i++) {
    printf("d18[%d] = %f\n", i, CPG_Result.dual->d18[i]);
  }
  for(i=0; i<13; i++) {
    printf("d19[%d] = %f\n", i, CPG_Result.dual->d19[i]);
  }
  for(i=0; i<4; i++) {
    printf("d20[%d] = %f\n", i, CPG_Result.dual->d20[i]);
  }
  for(i=0; i<4; i++) {
    printf("d21[%d] = %f\n", i, CPG_Result.dual->d21[i]);
  }
  for(i=0; i<13; i++) {
    printf("d22[%d] = %f\n", i, CPG_Result.dual->d22[i]);
  }
  for(i=0; i<4; i++) {
    printf("d23[%d] = %f\n", i, CPG_Result.dual->d23[i]);
  }
  for(i=0; i<4; i++) {
    printf("d24[%d] = %f\n", i, CPG_Result.dual->d24[i]);
  }
  for(i=0; i<13; i++) {
    printf("d25[%d] = %f\n", i, CPG_Result.dual->d25[i]);
  }
  for(i=0; i<4; i++) {
    printf("d26[%d] = %f\n", i, CPG_Result.dual->d26[i]);
  }
  for(i=0; i<4; i++) {
    printf("d27[%d] = %f\n", i, CPG_Result.dual->d27[i]);
  }
  for(i=0; i<13; i++) {
    printf("d28[%d] = %f\n", i, CPG_Result.dual->d28[i]);
  }
  for(i=0; i<4; i++) {
    printf("d29[%d] = %f\n", i, CPG_Result.dual->d29[i]);
  }
  for(i=0; i<4; i++) {
    printf("d30[%d] = %f\n", i, CPG_Result.dual->d30[i]);
  }

  return 0;

}
