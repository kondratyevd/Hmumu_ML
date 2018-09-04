
#ifndef GENMUON_HELPER
#define GENMUON_HELPER

#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/CommonIncludes.h"
#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/GenMuonInfo.h"
#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/GenParentInfo.h"

void FillGenMuonInfos( GenMuonInfos& _genMuonInfos, GenParentInfos& _genParentInfos,
		       const edm::Handle < reco::GenParticleCollection >& genPartons,
		       const bool isMC );

#endif  // #ifndef GENMUON_HELPER
