/*
 * MNCTModule.cxx
 *
 *
 * Copyright (C) 2008-2008 by Andreas Zoglauer.
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
// MNCTModule
//
////////////////////////////////////////////////////////////////////////////////


// Include the header:
#include "MNCTModule.h"

// Standard libs:

// ROOT libs:

// MEGAlib libs:

////////////////////////////////////////////////////////////////////////////////


#ifdef ___CINT___
ClassImp(MNCTModule)
#endif


////////////////////////////////////////////////////////////////////////////////


MNCTModule::MNCTModule()
{
  // Construct an instance of MNCTModule

  m_Name = "Base class...";
  m_XmlTag = "BaseClass"; // No spaces allowed

  m_HasOptionsGUI = false;
  
  m_IsOK = true;
  m_IsReady = true;
  
  m_Interrupt = false;
  
  m_Verbosity = 1;
}


////////////////////////////////////////////////////////////////////////////////


MNCTModule::~MNCTModule()
{
  // Delete this instance of MNCTModule
}


////////////////////////////////////////////////////////////////////////////////


bool MNCTModule::Initialize()
{
  //! Initialize the module --- has to be overwritten

  for (auto E: m_Expos) {
    E->Reset(); 
  }
  
  return true;
}
  

////////////////////////////////////////////////////////////////////////////////


MString MNCTModule::Report()
{
  return "";
}


////////////////////////////////////////////////////////////////////////////////


bool MNCTModule::ReadXmlConfiguration(MXmlNode* Node)
{
  //! Read the configuration data from an XML node

  return true;
}


////////////////////////////////////////////////////////////////////////////////


MXmlNode* MNCTModule::CreateXmlConfiguration() 
{
  //! Create an XML node tree from the configuration

  MXmlNode* Node = new MXmlNode(0, m_XmlTag);

  return Node;
}


// MNCTModule.cxx: the end...
////////////////////////////////////////////////////////////////////////////////
