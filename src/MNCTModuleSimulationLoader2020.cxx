/*
 * MNCTModuleSimulationLoader2020.cxx
 *
 *
 * Copyright (C) by Andreas Zoglauer.
 * All rights reserved.
 *
 *
 * This code implementation is the intellectual property of
 * Andreas Zoglauer.
 *
 * By copying, distributing or modifying the Program (or any work
 * based on the Program) you indicate your acceptance of this statement,
 * and all its terms.
 *
 */


////////////////////////////////////////////////////////////////////////////////
//
// MNCTModuleSimulationLoader2020
//
////////////////////////////////////////////////////////////////////////////////


// Include the header:
#include "MNCTModuleSimulationLoader2020.h"

// Standard libs:

// ROOT libs:
#include "TGClient.h"

// MEGAlib libs:
#include "MGUIOptionsSimulationLoader2020.h"
#include "MNCTModuleEventSaver.h"


////////////////////////////////////////////////////////////////////////////////


#ifdef ___CLING___
ClassImp(MNCTModuleSimulationLoader2020)
#endif


////////////////////////////////////////////////////////////////////////////////


MNCTModuleSimulationLoader2020::MNCTModuleSimulationLoader2020() : MModule()
{
  // Construct an instance of MNCTModuleSimulationLoader2020

  // Set the module name --- has to be unique
  m_Name = "Simulation loader and detector effects engine for COSI 2020";
  
  // Set the XML tag --- has to be unique --- no spaces allowed
  m_XmlTag = "XmlTagSimulationLoader2020";
  
  // This is a special start module which can generate its own events
  m_IsStartModule = true;
  
  // Allow the use of multiple threads and instances
  m_AllowMultiThreading = true;
  m_AllowMultipleInstances = false;

  // Set all modules, which have to be done before this module
  // None
  
  // Set all types this modules handles
  AddModuleType(MAssembly::c_EventLoader);
  AddModuleType(MAssembly::c_EventLoaderSimulation);
  AddModuleType(MAssembly::c_DetectorEffectsEngine);

  // Set all modules, which can follow this module
  AddSucceedingModuleType(MAssembly::c_NoRestriction);
  
  m_HasOptionsGUI = true;
}


////////////////////////////////////////////////////////////////////////////////


MNCTModuleSimulationLoader2020::~MNCTModuleSimulationLoader2020()
{
  // Delete this instance of MNCTModuleSimulationLoader2020
}


////////////////////////////////////////////////////////////////////////////////


bool MNCTModuleSimulationLoader2020::Initialize()
{
  // Initialize the module 

  MNCTDetectorEffectsEngineCOSI2020::SetGeometry(MModule::m_Geometry);
  if (MNCTDetectorEffectsEngineCOSI2020::Initialize() == false) {
    return false;
  }
  
  MSupervisor* S = MSupervisor::GetSupervisor();
  MNCTModuleEventSaver* Saver = dynamic_cast<MNCTModuleEventSaver*>(S->GetAvailableModuleByXmlTag("XmlTagEventSaver"));
  if (Saver != nullptr) {
    Saver->SetStartAreaFarField(m_StartAreaFarField);
  }
  
  return true;
}


////////////////////////////////////////////////////////////////////////////////


bool MNCTModuleSimulationLoader2020::AnalyzeEvent(MReadOutAssembly* Event) 
{
  // Main data analysis routine, which updates the event to a new level 

  if (MNCTDetectorEffectsEngineCOSI2020::GetNextEvent(Event) == false) {
    m_IsFinished = true;
    return false;
  }
    
  Event->SetAnalysisProgress(MAssembly::c_EventLoader | MAssembly::c_EventLoaderSimulation | MAssembly::c_DetectorEffectsEngine);
    
  return true;
}


////////////////////////////////////////////////////////////////////////////////


void MNCTModuleSimulationLoader2020::Finalize()
{
  // Initialize the module 

  MSupervisor* S = MSupervisor::GetSupervisor();
  MNCTModuleEventSaver* Saver = dynamic_cast<MNCTModuleEventSaver*>(S->GetAvailableModuleByXmlTag("XmlTagEventSaver"));
  if (Saver != nullptr) {
    Saver->SetSimulatedEvents(m_Reader->GetSimulatedEvents());
  }    
  
  MNCTDetectorEffectsEngineCOSI2020::Finalize();
}


///////////////////////////////////////////////////////////////////////////////


void MNCTModuleSimulationLoader2020::ShowOptionsGUI()
{
  //! Show the options GUI

  MGUIOptionsSimulationLoader2020* Options = new MGUIOptionsSimulationLoader2020(this);
  Options->Create();
  gClient->WaitForUnmap(Options);
}


////////////////////////////////////////////////////////////////////////////////


bool MNCTModuleSimulationLoader2020::ReadXmlConfiguration(MXmlNode* Node)
{
  //! Read the configuration data from an XML node
  
  MXmlNode* SimulationFileNameNode = Node->GetNode("SimulationFileName");
  if (SimulationFileNameNode != 0) {
    SetSimulationFileName(SimulationFileNameNode->GetValue());
  }
  MXmlNode* EnergyCalibrationFileNameNode = Node->GetNode("EnergyCalibrationFileName");
  if (EnergyCalibrationFileNameNode != 0) {
    SetEnergyCalibrationFileName(EnergyCalibrationFileNameNode->GetValue());
  }
  MXmlNode* DeadStripFileNameNode = Node->GetNode("DeadStripFileName");
  if (DeadStripFileNameNode != 0) {
    SetDeadStripFileName(DeadStripFileNameNode->GetValue());
  }
  MXmlNode* ThresholdFileNameNode = Node->GetNode("ThresholdFileName");
  if (ThresholdFileNameNode != 0) {
    SetThresholdFileName(ThresholdFileNameNode->GetValue());
  }
	MXmlNode* ChargeSharingFileNameNode = Node->GetNode("ChargeSharingFileNmae");
	if (ChargeSharingFileNameNode != 0) {
		SetChargeSharingFileName(ChargeSharingFileNameNode->GetValue());
	}
  MXmlNode* CrosstalkFileNameNode = Node->GetNode("CrosstalkFileName");
  if (CrosstalkFileNameNode != 0) {
    SetCrosstalkFileName(CrosstalkFileNameNode->GetValue());
  }
	MXmlNode* ChargeLossFileNameNode = Node->GetNode("ChargeLossFileName");
	if (ChargeLossFileNameNode != 0) {
		SetChargeLossFileName(ChargeLossFileNameNode->GetValue());
	}
  MXmlNode* DepthCalibrationCoeffsFileNameNode = Node->GetNode("DepthCalibrationCoeffsFileName");
  if (DepthCalibrationCoeffsFileNameNode != 0) {
    SetDepthCalibrationCoeffsFileName(DepthCalibrationCoeffsFileNameNode->GetValue());
  }
  MXmlNode* DepthCalibrationSplinesFileNameNode = Node->GetNode("DepthCalibrationSplinesFileName");
  if (DepthCalibrationSplinesFileNameNode != 0) {
    SetDepthCalibrationSplinesFileName(DepthCalibrationSplinesFileNameNode->GetValue());
  }
	MXmlNode* ApplyFudgeFactorNode = Node->GetNode("ApplyFudgeFactor");
	if (ApplyFudgeFactorNode != 0) {
		m_ApplyFudgeFactor = ApplyFudgeFactorNode->GetValueAsBoolean();
	}
 
  return true;
}


////////////////////////////////////////////////////////////////////////////////


MXmlNode* MNCTModuleSimulationLoader2020::CreateXmlConfiguration() 
{
  //! Create an XML node tree from the configuration
  
  MXmlNode* Node = new MXmlNode(0, m_XmlTag);  
  new MXmlNode(Node, "SimulationFileName", m_SimulationFileName);
  new MXmlNode(Node, "EnergyCalibrationFileName", m_EnergyCalibrationFileName);
  new MXmlNode(Node, "DeadStripFileName", m_DeadStripFileName);
  new MXmlNode(Node, "ThresholdFileName", m_ThresholdFileName);
	new MXmlNode(Node, "ChargeSharingFileNmae", m_ChargeSharingFileName);
	new MXmlNode(Node, "CrosstalkFileName", m_CrosstalkFileName);
	new MXmlNode(Node, "ChargeLossFileName", m_ChargeLossFileName);
  new MXmlNode(Node, "DepthCalibrationCoeffsFileName", m_DepthCalibrationCoeffsFileName);
  new MXmlNode(Node, "DepthCalibrationSplinesFileName", m_DepthCalibrationSplinesFileName);
  new MXmlNode(Node, "ApplyFudgeFactor", m_ApplyFudgeFactor);

  return Node;
}


// MNCTModuleSimulationLoader2020.cxx: the end...
////////////////////////////////////////////////////////////////////////////////