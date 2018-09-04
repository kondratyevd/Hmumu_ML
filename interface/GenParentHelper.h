
#ifndef GENPARENT_HELPER
#define GENPARENT_HELPER

#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/CommonIncludes.h"
#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/GenParentInfo.h"

void FillGenParentInfos( GenParentInfos& _genParentInfos,
                         const edm::Handle < reco::GenParticleCollection >& genPartons,
                         const std::vector<int> IDs, const bool isMC );

#endif  // #ifndef GENPARENT_HELPER
