#pragma once

#include <drogon/HttpController.h>

#include <stdio.h>
#include <string>

using namespace drogon;


class CustomCtrl : public drogon::HttpController<CustomCtrl, false>
{
  public:
    METHOD_LIST_BEGIN
        METHOD_ADD(CustomCtrl::hello, "/userName", Get);
    METHOD_LIST_END

    CustomCtrl(std::string inp){
        greeting = inp;
    }

    void hello(const HttpRequestPtr &req,
               std::function<void(const HttpResponsePtr &)> &&callback);


    private:
        std::string greeting;
};