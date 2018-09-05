
#ifndef ELE_PAIR_INFO
#define ELE_PAIR_INFO

#include <vector>
#include "TMath.h"

struct ElePairInfo {

  Int_t iEle1;
  Int_t iEle2;

  Double_t mass;
  // Double_t massErr;
  Double_t pt;
  Double_t eta;
  Double_t rapid;
  Double_t phi;
  Int_t    charge;
  Double_t dR;
  Double_t dEta;
  Double_t dPhi;
  Double_t dPhiStar;


  void init();

};

typedef std::vector<ElePairInfo> ElePairInfos;

#endif  // #ifndef ELE_PAIR_INFO