#pragma once

#include "pandabase.h"
#include "notifyCategoryProxy.h"
#include "configVariableDouble.h"
#include "configVariableString.h"
#include "configVariableInt.h"


NotifyCategoryDecl(mymodule, EXPORT_CLASS, EXPORT_TEMPL);

extern void init_libmymodule();
