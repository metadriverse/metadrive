
#include "config_module.h"

/*

Include all your dynamically typed classes here, e.g.
#include "my_dynamic_class.h"

*/

#include "dconfig.h"


Configure( config_mymodule );
NotifyCategoryDef( mymodule , "");

ConfigureFn( config_mymodule ) {
  init_libmymodule();
}

void
init_libmymodule() {
  static bool initialized = false;
  if (initialized) {
    return;
  }
  initialized = true;

  // Init your dynamic types here, e.g.:
  // MyDynamicClass::init_type();

  return;
}

