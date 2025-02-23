import { NextRequest, NextResponse } from "next/server";
import { PrismaClient } from '@prisma/client';
import prisma from '../../../util/connect'
/*
curl -H 'Content-Type: application/json' \
      -d '{ "classroomName":"Science1","student":"lalwani"}' \
      -X POST \
      http://localhost:3000/api/add-student */

// code for updating the classrooms
export const POST = async (req: NextRequest) => {
  try {
    const { classroomName, student } = await req.json();

    console.log(classroomName,  student);

    // const updatedAttempt = await prisma.attempt.updateMany({
    //     where: {
    //         userId: "004",
    //         id: "672c3d5ac11adb72d492e34c"
    //     },
    //     data:  {"level":"L1","rating":-1,"archetecture":"","lastLoss":-1}
    // });

    const classroom = await prisma.classrooms.findFirst({
        where: {
            // teacher: teacherEmail,
            classroomName: classroomName
        }
    })

    if (!classroom) {
        return NextResponse.json({ error: "Not Found" }, { status: 404 });
    }

    console.log(classroom);

    let curStudents = classroom.students 
    curStudents.push(student)

    const addStudent = await prisma.classrooms.updateMany({
        where: {
            classroomName: classroomName,
        },
        data: {
            students: curStudents
        }
    })

    if (addStudent.count == 0) {
      return NextResponse.json({ error: "Not Found" }, { status: 404 });
    }
    return NextResponse.json(
      { updatedClassroom: addStudent },
      { status: 200 }
    );
  } catch (error) {
    // console.error(error);
    return NextResponse.json({ error }, { status: 500 });
  }
};
