import { NextRequest, NextResponse } from "next/server";
import { PrismaClient } from '@prisma/client';
import prisma from '../../../util/connect'

/* curl -H 'Content-Type: application/json' \
      -d '{ "teacherEmail":"alawn2", "className":"ALan class"}' \
      -X POST \
      http://localhost:3000/api/create-classroom */

// code for updating the classrooms
export const POST = async (req: NextRequest) => {
  try {
    const { teacherEmail, className } = await req.json();

    // const updatedAttempt = await prisma.attempt.updateMany({
    //     where: {
    //         userId: "004",
    //         id: "672c3d5ac11adb72d492e34c"
    //     },
    //     data:  {"level":"L1","rating":-1,"archetecture":"","lastLoss":-1}
    // });

    console.log(teacherEmail)

    const addClassroom = await prisma.classrooms.create({
        data: {
            teacher: teacherEmail,
            classroomName: className,
            students: []
        }
    })

    if (addClassroom.count == 0) {
      return NextResponse.json({ error: "Not Found" }, { status: 404 });
    }
    return NextResponse.json(
      { updatedClassroom: addClassroom },
      { status: 200 }
    );
  } catch (error) {
    // console.error(error);
    return NextResponse.json({ error }, { status: 500 });
  }
};
