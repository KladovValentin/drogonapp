#include "TestCtrl.h"

using namespace std;

void CustomCtrl::hello(const HttpRequestPtr &req,
                       std::function<void(const HttpResponsePtr &)> &&callback)
{
    auto resp = HttpResponse::newHttpResponse();
    resp->setBody(greeting + " " + "sadsad");
    callback(resp);
}