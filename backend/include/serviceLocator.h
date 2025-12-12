#pragma once

#include "controllerBase.h"

class ServiceLocator{
public:
    static void initialize(std::shared_ptr<ControllerBase>& icontrollerBase) {
        instance().fcontrollerBase = icontrollerBase;
        cout << icontrollerBase->serverData << endl;
    }

    static ServiceLocator& instance() {
        static ServiceLocator instance;
        return instance;
    }

    std::shared_ptr<ControllerBase> getControllerBase() const {
        cout << fcontrollerBase->serverData << endl;
        return fcontrollerBase;
    }

private:
    std::shared_ptr<ControllerBase> fcontrollerBase;
};