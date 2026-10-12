/* 
 * TrappingCorrectionCs137.cxx
 *
 *
 * Copyright (C) by Sophie Haight.
 * All rights reserved.
 *
 *
 * This code implementation is the intellectual property of
 * Sophie Haight.
 *
 * By copying, distributing or modifying the Program (or any work
 * based on the Program) you indicate your acceptance of this statement,
 * and all its terms.
 *
 */


// Standard
#include <iostream>
#include <filesystem>
#include <string>
#include <sstream>
#include <fstream>
#include <csignal>
#include <cstdlib>
#include <map>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdio>
#include <cmath>
using namespace std;

// ROOT
#include <TROOT.h>
#include <TEnv.h>
#include <TSystem.h>
#include <TApplication.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TH2.h>
#include <TF1.h>
#include <TF2.h>
#include <TF1Convolution.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TGraph2D.h>
#include <TFitResultPtr.h>
#include <TFitResult.h>
#include <TStopwatch.h>
#include <TProfile.h>

// ROOT Fitting Engines
#include "Fit/Fitter.h"
#include "Fit/BinData.h"
#include "Fit/Chi2FCN.h"
#include "Math/WrappedMultiTF1.h"
#include "Math/Functor.h"

// MEGAlib
#include "MGlobal.h"
#include "MFile.h"
#include "MReadOutElementDoubleStrip.h"
#include "MFileReadOuts.h"
#include "MReadOutAssembly.h"
#include "MStripHit.h"
#include "MReadOutSequence.h"
#include "MSupervisor.h"
#include "MModuleLoaderMeasurementsHDF.h"
#include "MModuleEnergyCalibration.h"
#include "MModuleEventFilter.h"
#include "MModuleStripPairingMultiRoundChiSquare.h"
#include "MModuleStripPairingChiSquare.h"
#include "MModuleTACCalibration.h"
#include "MAssembly.h"
#include "MModuleDepthCalibration.h"


double g_MinCTD = -250;
// TO DO: Explore widening the CTD range 
double g_MaxCTD = 250;
double g_DetectorThickness = 15.0;
int g_MinCounts = 1000;
int g_HVStrips = 64;
int g_LVStrips = 64;

double g_CsPhotopeak = 661.7;

const int NCTDBins = 30;
// We need NCTDBins + 1 edges to define the boundaries of NCTDBins
double g_CTDBinEdges[NCTDBins + 1]; 

// Simulation Template Path for Joint Trapping Fitting
MString g_SimulatedCCEPath = "sim_cce_template.csv";

class TrappingCorrectionCs137
{
public:
  TrappingCorrectionCs137();
  ~TrappingCorrectionCs137();

  // Helper method to look up bin index for a given CTD value
  int GetCTDBin(double ctd) const;

  // Structs for CCE simulations & experimental points
  struct SimulatedCCE {
    std::vector<double> z_depth_mm;
    std::vector<double> e_cce_hv;
    std::vector<double> h_cce_hv;
    std::vector<double> e_cce_lv;
    std::vector<double> h_cce_lv;
  };

  struct ExperimentalPoint {
    int det_id;
    int ctd_bin;
    double ctd_midpoint_ns;
    double hv_cce;
    double hv_cce_err;
    double lv_cce;
    double lv_cce_err;
  };
  
  bool ParseCommandLine(int argc, char** argv);
  bool Analyze();
  void Interrupt() { m_Interrupt = true; }

  // Photopeak fitting function
  TF1* GeneratePhotopeakFunction();

  // Load Sim CCE files
  SimulatedCCE LoadSimulatedCCE(const std::string& sim_csv_path);
  
  // Output trapping paramters and simulated CCE insto csv file
  void SaveTrappingParamFile(const std::string& output_filepath,
                                 const std::string& trapping_param_filepath,
                                 const std::vector<double>& popt,
                                 const SimulatedCCE& sim,
                                 int detector_id);

 // Fit experimental CCE to simulated CCE to get trapping parameters
  void FitTrappingParameters(const std::vector<ExperimentalPoint>& exp_data,
                                     const std::string& sim_csv_path,
                                     const std::string& output_csv_path,
                                     const std::string& trapping_param_path,
                                     double detector_thickness,
                                     double default_ctd_range = 500.0,
                                     int detector_id = 0);

private:
    // Arrange CTD binning on sinusoidal distribution 
  void InitializeCTDBins();

  // CTD Range and Bins
  static const int NCTDBins = 30; // Set to your desired bin count
  double m_CTDBinEdges[NCTDBins + 1];
  double m_MinCTD = -250.0;
  double m_MaxCTD = 250.0;

  bool m_Interrupt;
  MString m_FileName;
  MString m_EcalFile;
  MString m_TACCalFile;
  MString m_StripMapFile;
  MString m_OutFile;
  
  bool m_PixelCorrect;
  bool m_MultiRoundStripPairing;
  bool m_ExcludeNN = true;
  bool m_ContinueHDF5; 

  double m_MinEnergy;
  double m_MaxEnergy;
};
////////////////////////////////////////////////////////////////////////////////

//! Constructor
TrappingCorrectionCs137::TrappingCorrectionCs137() : m_Interrupt(false)
{
  gStyle->SetPalette(1, 0);

  // Initialize non-linear CTD binning for the trapping pipeline
  InitializeCTDBins();
}

////////////////////////////////////////////////////////////////////////////////

//! Default destructor
TrappingCorrectionCs137::~TrappingCorrectionCs137()
{
  // Intentionally left blank
}

////////////////////////////////////////////////////////////////////////////////

//! Parse the command line
bool TrappingCorrectionCs137::ParseCommandLine(int argc, char** argv)
{
  ostringstream Usage;
  Usage<<endl;
  Usage<<"  Usage: TrappingCorrectionCs137 <options>"<<endl;
  Usage<<"    General options:"<<endl;
  Usage<<"         -i:   input file name (.hdf5 or .txt with list of hdf5s)"<<endl;
  Usage<<"         --emin:   minimum Event energy (default 600 keV)"<<endl;
  Usage<<"         --emax:   maximum Event energy (default 700 kev)"<<endl;
  Usage<<"         -e:   energy calibration file (.ecal)"<<endl;
  Usage<<"         --tcal:   TAC calibration file"<<endl;
  Usage<<"         -p:   do pixel-by-pixel correction"<<endl;
  Usage<<"         -m:   strip map file name (.map)"<<endl;
  Usage<<"         -g:   greedy strip pairing (default is chi-square)"<<endl;
  Usage<<"         -n:   exclude nearest neighbors"<<endl;
  Usage<<"         -ctdmin :   minimum CTD for binning (default -250 ns)"<<endl;
  Usage<<"         -ctdmax :   maximum CTD for binning (default 250 ns)"<<endl;
  Usage<<"         --thickness :   detector thickness in mm (default 15 mm)"<<endl;
  Usage<<"         --sim :   simulated CCE file (.csv)"<<endl;
  Usage<<"         -o:   outfile (default YYYYMMDDHHMMSS)"<<endl;
  Usage<<"         -h:   print this help"<<endl;
  Usage<<endl;

  string Option;

  // Check for help
  for (int i = 1; i < argc; i++) {
    Option = argv[i];
    if (Option == "-h" || Option == "--help" || Option == "?" || Option == "-?") {
      cout<<Usage.str()<<endl;
      return false;
    }
  }

  m_PixelCorrect = false;
  m_MultiRoundStripPairing = false; // should this be changed??
  m_ContinueHDF5 = true;
  m_MinEnergy = 600;
  m_MaxEnergy = 700;
  
  // Progress tracking 
  time_t rawtime;
  tm* timeinfo;
  time(&rawtime);
  timeinfo = gmtime(&rawtime);

  char buffer [80];
  strftime(buffer,80,"%Y%m%d%H%M%S",timeinfo);

  m_OutFile = MString(buffer);

  // Now parse the command line options:
  for (int i = 1; i < argc; i++) {
    Option = argv[i];

    // First check if each option has sufficient arguments:
    // Single argument
    if ((Option == "-i") || (Option == "-o") || (Option == "--emin") || (Option == "--emax") || (Option == "--tcal") || (Option == "-m") || (Option == "--ctdmin") || (Option == "--ctdmax") || (Option == "-thickness") || (Option == "--sim")) {
      if (!((argc > i+1) && (argv[i+1][0] != '-' || isalpha(argv[i+1][1]) == 0))){
        cout<<"Error: Option "<<argv[i][1]<<" needs a second argument!"<<endl;
        cout<<Usage.str()<<endl;
        return false;
      }
    }  

    /// Then fulfill the options:
    if (Option == "-i") {
      m_FileName = argv[++i];
      cout<<"Accepting file name: "<<m_FileName<<endl;
    }

    if (Option == "-e") {
      m_EcalFile = argv[++i];
      cout<<"Accepting file name: "<<m_EcalFile<<endl;
    } 

    if (Option == "--emin") {
      m_MinEnergy = stod(argv[++i]);
    } 

    if (Option == "--emax") {
      m_MaxEnergy = stod(argv[++i]);
    } 

    if (Option == "--tcal") {
      m_TACCalFile = argv[++i];
      cout<<"Accepting file name: "<<m_TACCalFile<<endl;
    } 

    if (Option == "-o"){
      m_OutFile = argv[++i];
      cout<<"Accepting file name: "<<m_OutFile<<endl;
    }

    if (Option == "-m"){
      m_StripMapFile = argv[++i];
      cout<<"Accepting file name: "<<m_StripMapFile<<endl;
    }

    if (Option == "--ctdmin") {
      g_MinCTD = stod(argv[++i]);
    } 

    if (Option == "--ctdmax") {
      g_MaxCTD = stod(argv[++i]);
    } 

    if (Option == "--thickness") {
      g_DetectorThickness = stod(argv[++i]);
    }

    if (Option == "--sim") {
      g_SimulatedCCEPath = argv[++i];
      cout<<"Accepting sim CCE file: "<<g_SimulatedCCEPath<<endl;
    }

    if (Option == "-p"){
      m_PixelCorrect = true;
    }

    if (Option == "-g"){
      m_MultiRoundStripPairing = true;
    }

    if (Option == "-n"){
      m_ExcludeNN = true;
    }
    if (Option == "--nocontinue") {
      m_ContinueHDF5 = false;
    } 

  }

  return true;
}


////////////////////////////////////////////////////////////////////////////////

//! Do whatever analysis is necessary
bool TrappingCorrectionCs137::Analyze()
{
  // time code just to see
  TStopwatch watch;
  watch.Start();

  if (m_Interrupt == true) return false;

  // [CTD][DetID][] HV Energy, LV Energy] for full detector values
  map<int, map<int, vector<double>>> FullDetEndpoints;

  // Store the input files
  vector<MString> FileNames;
  FileNames.push_back(m_FileName);
  cout << "file name stored" << endl;

  MString InputFile = FileNames[0];
  cout << " input file stored as: " << InputFile << endl;
  vector<MString> HDFNames;

  // Enable sum of weights squared for TH1 by default
  TH1::SetDefaultSumw2();

  // Detector-level maps organized by [CTDBin][DetID]
  map<int, map<int, TH1D*>> FullDetCTDHistograms;
  map<int, map<int, TH1D*>> FullDetHVEnergyHistograms;
  map<int, map<int, TH1D*>> FullDetLVEnergyHistograms;

  // Read in the input files and make a list of hdf5 files to calibrate
  if ((InputFile.GetSubString(InputFile.Length() - 4)) == "hdf5") {
    HDFNames.push_back(InputFile);
    cout << "hdf names loaded correctly" << endl;
  } else if ((InputFile.GetSubString(InputFile.Length() - 3)) == "txt") {
    cout << "Reading input file " << InputFile << endl;
    cout << "WARNING: When passing a list of files, ensure that you have chosen the correct HDF5 continuous reading mode. Use the --nocontinue option to suppress continuous file reading." << endl;
    MFile F;
    if (F.Open(InputFile) == false) {
      cout << "Error: Failed to open input file." << endl;
      return false;
    } else {
      MString Line;
      while (F.ReadLine(Line)) {
        MString Trimmed = Line.Trim();
        if ((Trimmed != "")) {
          if (F.Exists(Trimmed) == true) {
            HDFNames.push_back(Trimmed);
          } else {
            cout << "Error: Could not find file " << Trimmed << endl;
            return false;
          }
        }
      }
    }
  } else {
    cout << "Error: Unrecognized file format: " << InputFile << endl;
    return false;
  }

  // Analyze all the data and fill in the histograms
  for (unsigned int f = 0; f < HDFNames.size(); ++f) {

    MString File = HDFNames[f];
    cout << "Beginning analysis of file " << File << endl;

    // Create and initialize nuclearizer modules
    MSupervisor* S = MSupervisor::GetSupervisor();
    
    MModuleLoaderMeasurementsHDF* Loader;
    MModuleTACCalibration* TACCalibrator;
    MModuleEnergyCalibration* EnergyCalibrator;
    MModuleEventFilter* EventFilter;
    MModuleDepthCalibration* DepthCalibrator;

    unsigned int MNumber = 0;
    cout << "Creating HDF5 loader" << endl;
    Loader = new MModuleLoaderMeasurementsHDF();
    Loader->SetFileNameStripMap(m_StripMapFile);
    Loader->SetFileName(File);
    Loader->SetLoadContinuationFiles(m_ContinueHDF5);
    S->SetModule(Loader, MNumber);
    ++MNumber;

    cout << "Creating TAC calibrator" << endl;
    TACCalibrator = new MModuleTACCalibration();
    TACCalibrator->SetTACCalFileName(m_TACCalFile);
    S->SetModule(TACCalibrator, MNumber);
    ++MNumber;
    
    cout << "Creating energy calibrator" << endl;
    EnergyCalibrator = new MModuleEnergyCalibration();
    EnergyCalibrator->SetFileName(m_EcalFile);
    EnergyCalibrator->SetSlowThresholdCutMode(MSlowThresholdCutModes::e_Fixed);
    EnergyCalibrator->SetSlowThresholdCutFixedValue(30.0);
    S->SetModule(EnergyCalibrator, MNumber);
    ++MNumber;

    cout << "Creating Event filter" << endl;
    EventFilter = new MModuleEventFilter();
    EventFilter->SetMinimumLVStrips(1);
    EventFilter->SetMaximumLVStrips(3);
    EventFilter->SetMinimumHVStrips(1);
    EventFilter->SetMaximumHVStrips(3);
    EventFilter->SetMinimumHits(0);
    EventFilter->SetMaximumHits(100);
    EventFilter->SetMinimumTotalEnergy(m_MinEnergy);
    EventFilter->SetMaximumTotalEnergy(m_MaxEnergy ); // Multiply by 2 because this is the event-level energy, i.e. sum over both sides
    S->SetModule(EventFilter, MNumber);
    ++MNumber;
    
    cout << "Creating strip pairing" << endl;
    MModule* Pairing;
    if (m_MultiRoundStripPairing == true) {
      Pairing = new MModuleStripPairingMultiRoundChiSquare();
    } else {
      Pairing = new MModuleStripPairingChiSquare();
    }
    S->SetModule(Pairing, MNumber);

    // Create depth calibration module 
    // Only to import predefined functions, not for depth
    DepthCalibrator = new MModuleDepthCalibration();

    cout<<"Initializing Loader"<<endl;
    if (Loader->Initialize() == false) return false;
    cout<<"Initializing TAC calibrator"<<endl;
    if (TACCalibrator->Initialize() == false) return false;
    cout<<"Initializing Energy calibrator"<<endl;
    if (EnergyCalibrator->Initialize() == false) return false;
    cout<<"Initializing Event filter"<<endl;
    if (EventFilter->Initialize() == false) return false;
    cout<<"Initializing Pairing"<<endl;
    if (Pairing->Initialize() == false) return false;

    bool IsFinished = false;
    MReadOutAssembly* Event = new MReadOutAssembly();
    cout<<"Modules initialized, starting event loop"<<endl;

    cout << "Analyzing..." << endl;

  // --- PROGRESS TRACKING VARIABLES ---
  long long event_count = 0;
  long long processed_hits = 0;
  long long filled_events = 0;
  auto start_time = std::chrono::steady_clock::now();
  // -----------------------------------

  while ((IsFinished == false) && (m_Interrupt == false)) {
    Event->Clear();
    if (Loader->IsReady()) {

      Loader->AnalyzeEvent(Event);
      TACCalibrator->AnalyzeEvent(Event);
      EnergyCalibrator->AnalyzeEvent(Event);
      bool Unfiltered = EventFilter->AnalyzeEvent(Event);

      if (Unfiltered == true) {
        
        Pairing->AnalyzeEvent(Event);

        if ((Event->HasAnalysisProgress(MAssembly::c_StripPairing) == true) && (Unfiltered == true)) {
          
          for (unsigned int h = 0; h < Event->GetNHits(); ++h) {
            processed_hits++;
            double HVEnergy = 0.0;
            double LVEnergy = 0.0;
            vector<MStripHit*> HVStrips;
            vector<MStripHit*> LVStrips;

            MHit* H = Event->GetHit(h);
            int DetID = H->GetStripHit(0)->GetDetectorID();

            for (unsigned int sh = 0; sh < H->GetNStripHits(); ++sh) {
              MStripHit* SH = H->GetStripHit(sh);

              if ((m_ExcludeNN == false) || ((m_ExcludeNN == true) && (SH->IsNearestNeighbor() == false))) {
                if (SH->IsLowVoltageStrip() == true) {
                  LVEnergy += SH->GetEnergy();
                  LVStrips.push_back(SH);
                } else {
                  HVEnergy += SH->GetEnergy();
                  HVStrips.push_back(SH);
                }
              }
            }

              if ((HVStrips.size() > 0) && (LVStrips.size() > 0)) {
                
                double HVEnergyFraction = 0;
                double LVEnergyFraction = 0;
                MStripHit* HVSH = DepthCalibrator->GetDominantStrip(HVStrips, HVEnergyFraction); 
                MStripHit* LVSH = DepthCalibrator->GetDominantStrip(LVStrips, LVEnergyFraction);
                
                if ((LVSH != nullptr) && (HVSH != nullptr) && (LVSH->HasFastTiming() == true) && (HVSH->HasFastTiming() == true)) {
                  
                  double CTD = LVSH->GetTiming() - HVSH->GetTiming();
                  int CTDBin = GetCTDBin(CTD);
                  if (CTDBin < 0) continue;

                TH1D* FullHVHist  = FullDetHVEnergyHistograms[CTDBin][DetID];
                TH1D* FullLVHist  = FullDetLVEnergyHistograms[CTDBin][DetID];

                if (FullHVHist == nullptr) {
                  char name[128]; sprintf(name, "HV_Detector%d_bin%d", DetID, CTDBin);
                  FullHVHist = new TH1D(name, name, (m_MaxEnergy - m_MinEnergy) * 2, m_MinEnergy, m_MaxEnergy);
                  FullDetHVEnergyHistograms[CTDBin][DetID] = FullHVHist;
                }
                if (FullLVHist == nullptr) {
                  char name[128]; sprintf(name, "LV_Detector%d_bin%d", DetID, CTDBin);
                  FullLVHist = new TH1D(name, name, (m_MaxEnergy - m_MinEnergy) * 2, m_MinEnergy, m_MaxEnergy);
                  FullDetLVEnergyHistograms[CTDBin][DetID] = FullLVHist;
                }

                FullHVHist->Fill(HVEnergy);
                FullLVHist->Fill(LVEnergy);
                filled_events++;
              }
            }
          }
        }
      }
    }

    IsFinished = Loader->IsFinished();
  }

  // --- FINAL SUMMARY PRINT ---
  auto end_time = std::chrono::steady_clock::now();
  double total_time = std::chrono::duration<double>(end_time - start_time).count();

  cout << "\n==========================================" << endl;
  cout << " Finished Event Loop!" << endl;
  cout << " Total Events Processed: " << event_count << endl;
  cout << " Total Hits Examined:    " << processed_hits << endl;
  cout << " Total Histograms Filled:" << filled_events << endl;
  cout << " Time Elapsed:           " << total_time << " s" << endl;
  cout << " Exit Reason:            " << (m_Interrupt ? "User Interrupt (SIGINT)" : "Loader Finished") << endl;
  cout << "==========================================\n" << endl;

  // Store depth binned fit results in an output file 
  ofstream MasterFitFile;
  MasterFitFile.open(m_OutFile + MString("_All_CTDBin_FitResults.txt"));
  MasterFitFile << "======================================================================" << endl;
  MasterFitFile << "MASTER PHOTOPEAK FIT LOG FOR ALL CTD BINS AND DETECTORS" << endl;
  MasterFitFile << "======================================================================" << endl << endl;

  // Vector to hold experimental CCE points for joint fit
  std::vector<ExperimentalPoint> exp_points;

  // Do function fitting and recording for full detector outputs
  for (int c = 0; c < NCTDBins; ++c) {

    cout << "Processing CTD bin " << c << endl;
  
    for (auto const& [DetID, FullHVHist] : FullDetHVEnergyHistograms[c]) {
  
      TH1D* HVHist = FullDetHVEnergyHistograms[c][DetID];
      TH1D* LVHist = FullDetLVEnergyHistograms[c][DetID];

      if (HVHist->Integral() > g_MinCounts) {

        TF1* PhotopeakFunctionHV = GeneratePhotopeakFunction();
        TFitResultPtr HVFit = HVHist->Fit(PhotopeakFunctionHV, "S", "", 645, 675);

        TF1* PhotopeakFunctionLV = GeneratePhotopeakFunction();
        TFitResultPtr LVFit = LVHist->Fit(PhotopeakFunctionLV, "S", "", 645, 675);

        if ((HVFit >= 0) && (LVFit >= 0)) {
      
          // Clear or initialize the vector for this specific bin and detector
          FullDetEndpoints[c][DetID].clear();
          
          double hv_centroid = HVFit->Parameter(1);
          double hv_error    = HVFit->ParError(1);
          double lv_centroid = LVFit->Parameter(1);
          double lv_error    = LVFit->ParError(1);

          // Convert Centroids (keV) to CCE (dimensionless ratio relative to 661.7 keV)
          double hv_cce     = hv_centroid / g_CsPhotopeak;
          double hv_cce_err = hv_error / g_CsPhotopeak;
          double lv_cce     = lv_centroid / g_CsPhotopeak;
          double lv_cce_err = lv_error / g_CsPhotopeak;

          FullDetEndpoints[c][DetID].push_back(hv_cce);      // HV CCE
          FullDetEndpoints[c][DetID].push_back(hv_cce_err);  // HV CCE Err
          FullDetEndpoints[c][DetID].push_back(lv_cce);      // LV CCE
          FullDetEndpoints[c][DetID].push_back(lv_cce_err);  // LV CCE Err

          // Use double explicitly for the edge lookup and midpoint
          double edge_lower = static_cast<double>(m_CTDBinEdges[c]);
          double edge_upper = static_cast<double>(m_CTDBinEdges[c + 1]);
          double ctd_mid    = (edge_lower + edge_upper) / 2.0;

          // Ensure ctd_mid is non-zero when pushing to the experimental point vector
          exp_points.push_back({DetID, c, ctd_mid, hv_cce, hv_cce_err, lv_cce, lv_cce_err});

          MasterFitFile << "------------------------------------------------------------" << endl;
          MasterFitFile << " DETECTOR ID: " << DetID << "  |  CTD BIN INDEX: " << c << endl;
          MasterFitFile << "------------------------------------------------------------" << endl;
          
          // Redirect cout to our master file stream
          std::streambuf* coutbuf = cout.rdbuf();
          cout.rdbuf(MasterFitFile.rdbuf());
          
          // Restore normal terminal routing
          cout.rdbuf(coutbuf);

          MasterFitFile << endl << endl; // Add spacing between different bin entries

          TFile HVHistFile(m_OutFile + MString("_Det") + DetID + MString("_CTDbin_") + c + MString("_HVEnergyHist_Illum.root"), "recreate");
          TCanvas* HVHistCanvas = new TCanvas();
          HVHistCanvas->cd();
          HVHist->Draw("Hist");
          PhotopeakFunctionHV->Draw("same");
          HVHistCanvas->Write();
          HVHistFile.Close();

          TFile LVHistFile(m_OutFile + MString("_Det") + DetID + MString("_CTDbin_") + c + MString("_LVEnergyHist_Illum.root"), "recreate");
          TCanvas* LVHistCanvas = new TCanvas();
          LVHistCanvas->cd();
          LVHist->Draw("Hist");
          PhotopeakFunctionLV->Draw("same");
          LVHistCanvas->Write();
          LVHistFile.Close();

        } else {
          cout << "Fits failed for CTD bin " << c << " Detector " << DetID << endl;
        }
      } else {
        cout << "Fewer than " << g_MinCounts << " counts in CTD bin " << c << " Detector " << DetID << endl;
      }
    }
  }

  // Close output parameter file
  MasterFitFile.close();
  cout << "Master fit results log saved successfully." << endl;

  // Setup Experimental CCE parameter file
  ofstream OutputCalFile;
  OutputCalFile.open(m_OutFile + MString("_ExperimentalCCEs.txt"));
  
  // Header matching updated CCE outputs
  OutputCalFile << "Det_ID" << '\t' 
                << "CTD_Bin" << '\t' 
                << "CTD_BinMidpoint_ns" << '\t' 
                << "HV_CCE" << '\t'
                << "HV_CCE_error" << '\t'
                << "LV_CCE" << '\t'
                << "LV_CCE_error" << endl;
  cout << "Parameter file set up" << endl;

  // Loop systematically over each CTD bin first
  for (int c = 0; c < NCTDBins; ++c) {

    cout << "Writing output tracking parameters for CTD bin: " << c << endl;

    // Calculate the exact midpoint of this specific variable CTD bin
    double ctdBinMidpoint = (m_CTDBinEdges[c] + m_CTDBinEdges[c + 1]) / 2.0;

    // Loop over the detectors found inside this specific CTD bin
    for (auto const& [DetID, FitsVec] : FullDetEndpoints[c]) {
      
      double HV_CCE     = 0.0;
      double HV_CCEErr  = 0.0;
      double LV_CCE     = 0.0;
      double LV_CCEErr  = 0.0;

      if (FitsVec.size() >= 4) {
        HV_CCE    = FitsVec[0];
        HV_CCEErr = FitsVec[1];
        LV_CCE    = FitsVec[2];
        LV_CCEErr = FitsVec[3];
      }

      // Write row entries corresponding purely to the detector level measurements
      OutputCalFile << DetID << '\t'
                    << c << '\t'
                    << ctdBinMidpoint << '\t'
                    << HV_CCE << '\t' 
                    << HV_CCEErr << '\t' 
                    << LV_CCE << '\t'
                    << LV_CCEErr << endl;
    }
  }

  OutputCalFile.close();
  cout << "Experimental CCE file saved successfully." << endl;
  
  // ---- Fit experimental CCE to simulated CCE to extract trapping parameters for each detector ----
  cout << "\nStarting Joint Trapping Fitting and Matrix Generation..." << endl;
  for (int det_id = 0; det_id < 4; ++det_id) {
    std::string out_csv   = "Det_" + std::to_string(det_id) + "_sim_output.csv";
    std::string param_csv = "Det_" + std::to_string(det_id) + "_trapping_params_" + std::to_string(NCTDBins) + "bins.csv";

    // Filter experimental points for current detector
    std::vector<ExperimentalPoint> det_exp_points;
    for (const auto& pt : exp_points) {
      if (pt.det_id == det_id) {
        det_exp_points.push_back(pt);
      }
    }

    // STOP if no points were successfully extracted for this detector!
    if (det_exp_points.empty()) {
      std::cerr << "[Error] No valid experimental points extracted for Detector " 
                << det_id << ". Skipping joint fit." << std::endl;
      continue; // Skip to next detector
    }

    FitTrappingParameters(det_exp_points, g_SimulatedCCEPath.Data(), out_csv, param_csv, g_DetectorThickness, (g_MaxCTD - g_MinCTD) , det_id);
  }

  watch.Stop();
  cout << "total time (s): " << watch.CpuTime() << endl;
 
  return true;
  }
}

////////////////////////////////////////////////////////////////////////////////////

TF1* TrappingCorrectionCs137::GeneratePhotopeakFunction()
{
  // Component 1: Core Gaussian
  // exp(-(x-x0)^2 / (2*sigma^2))
  MString gaussStr = "exp(-(x-[1])^2 / (2*[2]^2))";

  // Component 2: Exponential Tail + Shelf
  // BoverA * exp(gamma*(x-x0)) * 0.5 * erfc((x-x0)/(sigma*sigma_ratio*sqrt(2)))
  MString expTailStr = "[3] * exp([4]*(x-[1])) * 0.5 * erfc((x-[1])/([2]*[5]*sqrt(2)))";

  // Component 3: Linear Tail + Shelf
  // BoverA * CoverB * (1 + D*(x-x0)) * 0.5 * erfc((x-x0)/(sigma*sigma_ratio*sqrt(2)))
  MString linTailStr = "[3] * [6] * (1 + [7]*(x-[1])) * 0.5 * erfc((x-[1])/([2]*[5]*sqrt(2)))";

  // Combine components with an overall normalization scaling factor [0]
  MString fullFormula = "[0] * (" + gaussStr + " + " + expTailStr + " + " + linTailStr + ")";

  // Instantiate TF1 over your expected fit window
  TF1* PhotopeakFunction = new TF1("PhotopeakFunction", fullFormula.Data(), 645, 675);

  // Set Parameter Names
  PhotopeakFunction->SetParName(0, "Amplitude");
  PhotopeakFunction->SetParName(1, "x0 (Mu)");
  PhotopeakFunction->SetParName(2, "Sigma Gauss");
  PhotopeakFunction->SetParName(3, "BoverA");
  PhotopeakFunction->SetParName(4, "Gamma");
  PhotopeakFunction->SetParName(5, "Sigma Ratio");
  PhotopeakFunction->SetParName(6, "CoverB");
  PhotopeakFunction->SetParName(7, "D (Lin Slope)");

  // Provide initial sensible guesses for a Cs137 photopeak
  PhotopeakFunction->SetParameter("Amplitude", 1000);
  PhotopeakFunction->SetParameter("x0 (Mu)", 661.7);
  PhotopeakFunction->SetParameter("Sigma Gauss", 2.0);
  PhotopeakFunction->SetParameter("BoverA", 0.05);
  PhotopeakFunction->SetParameter("Gamma", 0.5);
  PhotopeakFunction->SetParameter("Sigma Ratio", 0.85);
  PhotopeakFunction->SetParameter("CoverB", 0.13);
  PhotopeakFunction->SetParameter("D (Lin Slope)", 0.028);

  // Set boundary limits to stabilize convergence
  PhotopeakFunction->SetParLimits(0, 1, 1e8);
  PhotopeakFunction->SetParLimits(1, 645, 675);     // Keeps peak centered around 662 keV
  PhotopeakFunction->SetParLimits(2, 0.5, 10);      // Prevents sigma from blowing up or hitting zero
  PhotopeakFunction->SetParLimits(3, 0.0, 1.0);     // Tail shouldn't be larger than the main peak
  PhotopeakFunction->SetParLimits(4, 0.001, 2.0);   // Standard range for exponential decay factor
  PhotopeakFunction->SetParLimits(5, 0.1, 5.0);     // Ratio of shelf width to peak width
  PhotopeakFunction->SetParLimits(6, 0.0, 5.0);     
  PhotopeakFunction->SetParLimits(7, -1.0, 1.0);
  // // Gaussian with a low-E shelf
  // TF1* PhotopeakFunction = new TF1("PhotopeakFunction", "gaus(0) + [0]*[3]*(1 - erf((x-[1])/(sqrt(2)*[2])))", 620, 680);

  // PhotopeakFunction->SetParName(0, "Gauss norm");
  // PhotopeakFunction->SetParName(1, "Mu");
  // PhotopeakFunction->SetParName(2, "Sigma");
  // PhotopeakFunction->SetParName(3, "Shelf norm");

  // PhotopeakFunction->SetParameter("Gauss norm", 1000);
  // PhotopeakFunction->SetParameter("Mu", 661.7);
  // PhotopeakFunction->SetParameter("Sigma", 2);
  // PhotopeakFunction->SetParameter("Shelf norm", 0.05);

  // PhotopeakFunction->SetParLimits(0, 10, 1e8);
  // PhotopeakFunction->SetParLimits(1, 652, 672);
  // PhotopeakFunction->SetParLimits(2, 1.0, 10);
  // PhotopeakFunction->SetParLimits(3, 0, 0.1);

  return PhotopeakFunction;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

TrappingCorrectionCs137::SimulatedCCE TrappingCorrectionCs137::LoadSimulatedCCE(const std::string& csv_path) {
  SimulatedCCE sim;
  std::ifstream file(csv_path);

  if (!file.is_open()) {
    std::cerr << "[Error] Could not open simulation CSV at: " << csv_path << std::endl;
    return sim;
  }

  std::string line;
  while (std::getline(file, line)) {
    // Skip empty lines, comment lines starting with #, or header text
    if (line.empty() || line[0] == '#' || line.find("z_depth") != std::string::npos) {
      continue;
    }

    std::stringstream ss(line);
    std::string val;
    std::vector<double> row;

    while (std::getline(ss, val, ',')) {
      try {
        row.push_back(std::stod(val));
      } catch (...) {
        row.clear();
        break; // Skip non-numeric header lines
      }
    }

    // Expecting 5 columns: [z_depth_cm, e_CCE_HV, h_CCE_HV, e_CCE_LV, h_CCE_LV]
    if (row.size() == 5) {
      
      double z_mm = row[0]; 

      sim.z_depth_mm.push_back(z_mm);
      sim.e_cce_hv.push_back(row[1]);
      sim.h_cce_hv.push_back(row[2]);
      sim.e_cce_lv.push_back(row[3]);
      sim.h_cce_lv.push_back(row[4]);
    }
  }

  return sim;
}
////////////////////////////////////////////////////////////////////////////////////////////////////

void TrappingCorrectionCs137::SaveTrappingParamFile(const std::string& output_filepath,
                                 const std::string& trapping_param_filepath,
                                 const std::vector<double>& popt,
                                 const SimulatedCCE& sim,
                                 int detector_id) 
{
  // Define trapping parameters 
  double A_hv = popt[0];
  double A_lv = popt[1];
  double B    = popt[2];
  double C    = popt[3];

  // Lambda function to write CSV content
  auto write_file_content = [&](const std::string& filepath) {
    std::ofstream csv_file(filepath, std::ios::out | std::ios::trunc);
    if (!csv_file.is_open()) {
      std::cerr << "[Error] Cannot create output file: " << filepath << std::endl;
      return;
    }

    //  Detector ID
    csv_file << "### " << detector_id << "\n";
    
    // Trapping parameters
    csv_file << A_hv << "," << A_lv << "," << B << "," << C << "\n";
    
    // Column Headers 
    csv_file << "# z_depth_cm,e_CCE_HV,h_CCE_HV,e_CCE_LV,h_CCE_LV\n";
    
    // Array Data (Converted Depth from mm to cm)
    for (size_t i = 0; i < sim.z_depth_mm.size(); ++i) {
      double sim_z_cm = sim.z_depth_mm[i] / 10.0; // sims are in mm but nuclearizer expects cm
      csv_file << sim_z_cm << ","
               << sim.e_cce_hv[i] << ","
               << sim.h_cce_hv[i] << ","
               << sim.e_cce_lv[i] << ","
               << sim.h_cce_lv[i] << "\n";
    }
    csv_file.close();
  };

  write_file_content(output_filepath);
  write_file_content(trapping_param_filepath);

  std::cout << "Successfully generated trapping parameter file for Detector " 
            << detector_id << " at: " << output_filepath << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TrappingCorrectionCs137::FitTrappingParameters(const std::vector<ExperimentalPoint>& exp_data,
  const std::string& sim_csv_path,
  const std::string& output_csv_path,
  const std::string& trapping_param_path,
  double detector_thickness,
  double default_ctd_range,
  int detector_id) 
{
// Load Simulation Template
SimulatedCCE sim = LoadSimulatedCCE(sim_csv_path);

if (sim.z_depth_mm.empty()) {
std::cerr << "[FATAL] Simulation template is empty or file not found at: " 
<< sim_csv_path << std::endl;
return;
}

/////////////// Data Filters /////////////////////

// Reject data points with high error or zero error
const double max_allowed_cce_err = 6.0 / g_CsPhotopeak;  

auto hv_ok = [&](const ExperimentalPoint& p) {
  return p.hv_cce_err > 0.0 && p.hv_cce_err <= max_allowed_cce_err;
};
auto lv_ok = [&](const ExperimentalPoint& p) {
  return p.lv_cce_err > 0.0 && p.lv_cce_err <= max_allowed_cce_err;
};

std::vector<ExperimentalPoint> det_exp_data;
int rejected_counts = 0, n_hv = 0, n_lv = 0;

// Only retain HV or LV data points w acceptable errors
for (const auto& pt : exp_data) {
  if (pt.det_id != detector_id && detector_id != -1) continue;
  const bool h = hv_ok(pt), l = lv_ok(pt);
  if (!h && !l) { rejected_counts++; continue; }   // neither side usable
  det_exp_data.push_back(pt);
  if (h) n_hv++;
  if (l) n_lv++;
}

if (det_exp_data.empty()) {
std::cout << "[Info] No valid data points left after error filtering for Detector " 
<< detector_id << std::endl;
return;
}

// Set up and sort 1D Interpolation Graphs for Sim CCEs
TGraph g_hv_e(sim.z_depth_mm.size(), sim.z_depth_mm.data(), sim.e_cce_hv.data());
TGraph g_hv_h(sim.z_depth_mm.size(), sim.z_depth_mm.data(), sim.h_cce_hv.data());
TGraph g_lv_e(sim.z_depth_mm.size(), sim.z_depth_mm.data(), sim.e_cce_lv.data());
TGraph g_lv_h(sim.z_depth_mm.size(), sim.z_depth_mm.data(), sim.h_cce_lv.data());

g_hv_e.Sort(); g_hv_h.Sort();
g_lv_e.Sort(); g_lv_h.Sort();

double z_min_sim = sim.z_depth_mm.front();
double z_max_sim = sim.z_depth_mm.back();

// Depth conversion factor (CTD ns -> Depth mm)
double ctd_to_mm_factor = (detector_thickness / 2.0) / -(default_ctd_range / 2.0);

// Chi2 cost fuction to fit trapping parameters
auto chi2_lambda = [&](const double* p) -> double {
  double chi2 = 0.0;
  
  double A_hv = p[0];
  double A_lv = p[1];
  double B    = p[2];
  double C    = p[3];

  for (const auto& pt : det_exp_data) {
    double z = pt.ctd_midpoint_ns * ctd_to_mm_factor;

    // If the measured depth is outide of simulation bounds assign it the closest CCE w/i the simulation
    if (z < z_min_sim) z = z_min_sim;
    if (z > z_max_sim) z = z_max_sim;

    // Evaluate TGraphs
    double e_hv = g_hv_e.Eval(z);
    double h_hv = g_hv_h.Eval(z);
    double e_lv = g_lv_e.Eval(z);
    double h_lv = g_lv_h.Eval(z);


    double pred_hv = A_hv * (1.0 - B * (1.0 - e_hv)) * (1.0 - C * (1.0 - h_hv));
    double pred_lv = A_lv * (1.0 - B * (1.0 - e_lv)) * (1.0 - C * (1.0 - h_lv));

    //DEBUG: add systematic error to help fitter converge
    double g_SysErrKeV = 0.0;  // Systematic error in keV, can be adjusted based on calibration
    const double sys_cce = g_SysErrKeV / g_CsPhotopeak;   // new option, e.g. 0.2 keV
    double err_hv = std::hypot(pt.hv_cce_err, sys_cce);
    double err_lv = std::hypot(pt.lv_cce_err, sys_cce);

    // For valid data points, find chi2
    if (hv_ok(pt)) {
      double d = (pt.hv_cce - pred_hv) / err_hv;
      chi2 += d * d;
    }
    if (lv_ok(pt)) {
      double d = (pt.lv_cce - pred_lv) / err_lv;
      chi2 += d * d;
    }
  }
  
  return std::isnan(chi2) ? 1e9 : chi2;
};

//DEBUG
double py_best[4] = {1.00143528, 1.00046766, 1.0786083, 0.29929985};
std::cout << "chi2 at python optimum = " << chi2_lambda(py_best) << std::endl;   // expect ≈ 781.4

// Wrap lambda inside ROOT::Math::Functor
ROOT::Math::Functor fcn(chi2_lambda, 4);

// Configure Fitter
ROOT::Fit::Fitter fitter;

// Initialize parameters and steps 
double init_params[4] = {1.0, 1.0, 0.1, 0.1};;
double init_steps[4]  = {1e-4, 1e-4, 0.05, 0.05};

fitter.Config().SetMinimizer("Minuit2", "Migrad");
fitter.Config().MinimizerOptions().SetStrategy(2);
fitter.Config().MinimizerOptions().SetTolerance(1e-4);
fitter.Config().MinimizerOptions().SetMaxFunctionCalls(100000);
fitter.Config().SetParamsSettings(4, init_params, init_steps);

fitter.Config().ParSettings(0).SetLimits(0.9, 1.1);
fitter.Config().ParSettings(1).SetLimits(0.9, 1.1);
fitter.Config().ParSettings(2).SetLimits(0.0, 5.0);
fitter.Config().ParSettings(3).SetLimits(0.0, 2.0);

fitter.SetFCN(fcn);
// // bool ok = fitter.FitFCN();
// fitter.Result().Print(std::cout);   // check status/EDM/covariance

//DEBUG??

///////////////////////// Printed fit statistics /////////////////////////////////////////

// Extract Chi2 manually if fitter result still reports -1
const ROOT::Fit::FitResult& result = fitter.Result();
std::vector<double> popt(4, 0.0);
for (int i = 0; i < 4; ++i) popt[i] = result.Parameter(i);

double calculated_chi2 = chi2_lambda(popt.data());
int n_points = n_hv + n_lv; // Total data points (HV + LV)
int ndf = n_points - 4;

std::cout << "\n================ JOINT FIT RESULTS ================" << std::endl;
std::cout << "  Detector ID : " << detector_id << std::endl;
std::cout << "  A_hv        : " << popt[0] << std::endl;
std::cout << "  A_lv        : " << popt[1] << std::endl;
std::cout << "  B (e-trap)  : " << popt[2] << std::endl;
std::cout << "  C (h-trap)  : " << popt[3] << std::endl;
std::cout << "  Chi2 (Calc) : " << calculated_chi2 << std::endl;
std::cout << "  NDF         : " << ndf << std::endl;
std::cout << "  Chi2 / NDF  : " << (ndf > 0 ? calculated_chi2 / ndf : -1) << std::endl;
std::cout << "====================================================\n" << std::endl;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
SaveTrappingParamFile(output_csv_path, trapping_param_path, popt, sim, detector_id);
}

///////////////////////////////////////////////////////////////////////////////////

void TrappingCorrectionCs137::InitializeCTDBins() {
  double center = (m_MaxCTD + m_MinCTD) / 2.0;
  double halfWidth = (m_MaxCTD - m_MinCTD) / 2.0;

  for (int i = 0; i <= NCTDBins; ++i) {
    double fraction = -1.0 + 2.0 * double(i) / double(NCTDBins);
    double nonLinearFraction = std::sin(fraction * M_PI / 2.0); 
    m_CTDBinEdges[i] = center + halfWidth * nonLinearFraction;
  }
}

//////////////////////////////////////////////////////////////////////////////////

int TrappingCorrectionCs137::GetCTDBin(double ctd) const 
{
  // Hard bounds check against member bounds
  if (ctd < m_MinCTD || ctd >= m_MaxCTD) return -1;

  // Binary search on class member array m_CTDBinEdges
  auto it = std::upper_bound(m_CTDBinEdges, m_CTDBinEdges + NCTDBins + 1, ctd);
  
  // Calculate bin index from iteration offset
  int bin = std::distance(m_CTDBinEdges, it) - 1;

  // Guard against edge cases at boundaries
  if (bin >= NCTDBins) bin = NCTDBins - 1;
  if (bin < 0) bin = 0;

  return bin;
}

////////////////////////////////////////////////////////////////////////////////


TrappingCorrectionCs137* g_Prg = 0;
int g_NInterruptCatches = 1;

////////////////////////////////////////////////////////////////////////////////


//! Called when an interrupt signal is flagged
//! All catched signals lead to a well defined exit of the program
void CatchSignal(int a)
{
  if (g_Prg != 0 && g_NInterruptCatches-- > 0) {
    cout<<"Catched signal Ctrl-C (ID="<<a<<"):"<<endl;
    g_Prg->Interrupt();
  } else {
    abort();
  }
}


////////////////////////////////////////////////////////////////////////////////


//! Main program
int main(int argc, char** argv)
{
  // Catch a user interrupt for graceful shutdown
  signal(SIGINT, CatchSignal);

  // Initialize global MEGAlib variables
  MGlobal::Initialize("TrappingCorrectionCs137", "A standalone program to determine CTD-dependent shifts of the Cs137 peak");

  TApplication TrappingCorrectionApp("TrappingCorrectionApp", 0, 0);

  // Instantiates class and runs InitializeCTDBins() automatically
  g_Prg = new TrappingCorrectionCs137();

  if (g_Prg->ParseCommandLine(argc, argv) == false) {
    cerr << "Error during parsing of command line!" << endl;
    return -1;
  } 
  if (g_Prg->Analyze() == false) {
    cerr << "Error during analysis!" << endl;
    return -2;
  } 

  TrappingCorrectionApp.Run();

  cout << "Program exited normally!" << endl;

  return 0;
}

////////////////////////////////////////////////////////////////////////////////