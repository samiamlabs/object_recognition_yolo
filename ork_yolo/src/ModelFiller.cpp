#include <ecto/ecto.hpp>

#include <object_recognition_core/common/types.h>
#include <object_recognition_core/db/document.h>

using object_recognition_core::db::ObjectId;
using object_recognition_core::db::Document;

namespace yolo_ecto
{
  struct ModelFiller
  {
  public:
    static void
    declare_io(const ecto::tendrils& params, ecto::tendrils& inputs, ecto::tendrils& outputs)
    {
    }

    int
    process(const ecto::tendrils& inputs, const ecto::tendrils& outputs)
    {
      return ecto::OK;
    }

  private:

  };
}

ECTO_CELL(ecto_yolo, yolo_ecto::ModelFiller, "ModelFiller",
          "Populates a db document with a YOLO model for persisting a later date.")
