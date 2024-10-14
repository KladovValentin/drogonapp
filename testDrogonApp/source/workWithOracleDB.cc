#include "../include/workWithOracleDB.h"

#include "../include/functions.h"
#include "../include/constants.h"

#include <iomanip>

using namespace std;

using namespace dateRunF;
namespace fs = std::filesystem;

OracleDBManager::OracleDBManager(){
    connectToDB();
}
OracleDBManager::~OracleDBManager(){}


// Error checking function
void OracleDBManager::checkerr(OCIError *errhp, sword status) {
    if (status != OCI_SUCCESS) {
        text errbuf[512];
        sb4 errcode = 0;
        OCIErrorGet((dvoid *)errhp, (ub4)1, (text *)NULL, &errcode,
                    errbuf, (ub4)sizeof(errbuf), OCI_HTYPE_ERROR);
        std::cerr << "OCI Error: " << errbuf << std::endl;
        exit(1);
    }
}



void OracleDBManager::connectToDB(){
    // Initialize the OCI environment
    OCIInitialize((ub4) OCI_DEFAULT, (dvoid *) 0, (dvoid * (*)(dvoid *, size_t)) 0,
                  (dvoid * (*)(dvoid *, dvoid *, size_t)) 0, (void (*)(dvoid *, dvoid *)) 0);

    OCIEnvInit((OCIEnv **) &envhp, OCI_DEFAULT, (size_t) 0, (dvoid **) 0);

    OCIHandleAlloc((dvoid *) envhp, (dvoid **) &errhp, OCI_HTYPE_ERROR, (size_t) 0, (dvoid **) 0);

    OCIHandleAlloc((dvoid *) envhp, (dvoid **) &svchp, OCI_HTYPE_SVCCTX, (size_t) 0, (dvoid **) 0);


    // Connect to the database
    const char* user = "DAQ_PUB";
    const char* pass = "hades";
    const char* service = "dbi:Oracle:db-hades";
    cout << "connecting...." << endl;
    if (OCILogon(envhp, errhp, &svchp, (text *) user, strlen(user),
             (text *) pass, strlen(pass),
             (text *) service, strlen(service))) {
             //(text *) "", 0) != OCI_SUCCESS) { 
        checkerr(errhp, OCI_ERROR);
    }
    else {
        cout << "connected?" << endl;
    }

    // Execute the SQL statement
    //checkerr(errhp, OCIStmtExecute(svchp, stmthp, errhp, (ub4) 0, (ub4) 0, (OCISnapshot *) NULL, (OCISnapshot *) NULL, OCI_DEFAULT));
    
      // Prepare the SQL statement to list service names
    const char *sql = "SELECT name FROM v$services";

    OCIHandleAlloc((dvoid *) envhp, (dvoid **) &stmthp, OCI_HTYPE_STMT, (size_t) 0, (dvoid **) 0);
    checkerr(errhp, OCIStmtPrepare(stmthp, errhp, (text *) sql,
                                   (ub4) strlen(sql), (ub4) OCI_NTV_SYNTAX, (ub4) OCI_DEFAULT));

    checkerr(errhp, OCIStmtExecute(svchp, stmthp, errhp, (ub4) 0, (ub4) 0, (OCISnapshot *) NULL, (OCISnapshot *) NULL, OCI_DEFAULT));

    // Fetch the results
    char service_name[128];
    checkerr(errhp, OCIDefineByPos(stmthp, &defnp, errhp, 1, (dvoid *) &service_name,
                                  (sb4) sizeof(service_name), SQLT_STR, (dvoid *) 0, (ub2 *) 0, (ub2 *) 0, OCI_DEFAULT));

    while (OCIStmtFetch(stmthp, errhp, 1, OCI_FETCH_NEXT, OCI_DEFAULT) == OCI_SUCCESS) {
        std::cout << "Service Name: " << service_name << std::endl;
    }


    /*
    // Get the number of columns
    ub4 col_count;
    OCIAttrGet((dvoid *) stmthp, (ub4) OCI_HTYPE_STMT, (dvoid *) &col_count, (ub4 *) 0, (ub4) OCI_ATTR_PARAM_COUNT, errhp);

    // Iterate over the columns to get their labels and descriptions
    for (ub4 i = 1; i <= col_count; i++) {
        OCIParam *colhp;
        text *col_name;
        ub4 col_name_len;
        ub2 data_type;

        // Get the column descriptor
        OCIParamGet((dvoid *) stmthp, (ub4) OCI_HTYPE_STMT, errhp, (dvoid **) &colhp, i);

        // Get the column name
        OCIAttrGet((dvoid *) colhp, (ub4) OCI_DTYPE_PARAM, (dvoid **) &col_name, (ub4 *) &col_name_len, (ub4) OCI_ATTR_NAME, errhp);

        // Get the data type of the column
        OCIAttrGet((dvoid *) colhp, (ub4) OCI_DTYPE_PARAM, (dvoid *) &data_type, (ub4 *) 0, (ub4) OCI_ATTR_DATA_TYPE, errhp);

        // Print the column name and data type
        std::cout << "Column " << i << ": " << std::string((char *) col_name, col_name_len) << ", Data Type: " << data_type << std::endl;
    }*/

    // Clean up
    OCIHandleFree((dvoid *) stmthp, OCI_HTYPE_STMT);
    OCILogoff(svchp, errhp);
    OCIHandleFree((dvoid *) svchp, OCI_HTYPE_SVCCTX);
    OCIHandleFree((dvoid *) errhp, OCI_HTYPE_ERROR);
    OCIHandleFree((dvoid *) envhp, OCI_HTYPE_ENV);
}



int OracleDBManager::getDBdataBasic(){


    return 0;
}